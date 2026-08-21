# -*- coding: utf-8 -*-
"""
PDF → Markdown 转换工具(知识库构建前置步骤)

两种方式:
1. MinerU API(推荐,高质量):本地 PDF 上传到 mineru.net → 轮询 → 下载 full.md。
   需在 config.yaml 填写 MINERU_API_KEY。
2. pypdf 兜底(无 key 时):用 pypdf 按页抽取文本,写入 "# Page N" 标记,
   供 LlamaIndex text_splitter 按页切分。质量较差(无表格/排版),仅用于快速构建/测试。

约定:输出 .md 文件名与 PDF 同名(仅改扩展名),存入 config.RAG_MD_DIR。

可作为模块导入调用,也可独立运行:
    python tools/pdf_to_md.py            # 转换 RAG_PDF_DIR 下所有 PDF
    python tools/pdf_to_md.py --convert   # 强制重新转换(覆盖已有 MD)
"""
import logging
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import List, Optional

# 项目根入 sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import compat  # noqa: E402

import requests

import config

logger = logging.getLogger("mask_agent.pdf2md")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)
logger.propagate = False

_MINERU_BASE = "https://mineru.net/api/v4"


# ============================================================
# MinerU API 方式
# ============================================================
def _mineru_headers() -> dict:
    key = (getattr(config, "MINERU_API_KEY", "") or "").strip()
    if not key:
        raise ValueError("未配置 MINERU_API_KEY,请在 config.yaml 填写")
    return {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}


def _apply_upload_url(file_name: str) -> tuple:
    """申请预签名上传链接(POST /file-urls/batch),返回 (batch_id, file_urls)。"""
    url = f"{_MINERU_BASE}/file-urls/batch"
    data = {
        "files": [{"name": file_name, "is_ocr": bool(getattr(config, "MINERU_IS_OCR", False))}],
        "enable_formula": bool(getattr(config, "MINERU_ENABLE_FORMULA", True)),
    }
    res = requests.post(url, headers=_mineru_headers(), json=data, timeout=60)
    result = res.json()
    if result.get("code") != 0:
        raise RuntimeError(f"申请上传链接失败: {result.get('msg')}(code={result.get('code')})")
    return result["data"]["batch_id"], result["data"]["file_urls"]


def _upload_file(local_path: Path, upload_url: str) -> None:
    """PUT 上传本地文件到预签名 URL(不带任何 header,OSS 签名已内嵌 URL)。"""
    with open(local_path, "rb") as f:
        res = requests.put(upload_url, data=f, timeout=300)
    if res.status_code != 200:
        raise RuntimeError(f"文件上传失败: HTTP {res.status_code} - {res.text[:200]}")


def _download_and_unzip(full_zip_url: str, batch_id: str, download_dir: str) -> str:
    local_filename = os.path.join(download_dir, f"{batch_id}.zip")
    logger.info("开始下载: %s", full_zip_url)
    r = requests.get(full_zip_url, stream=True, timeout=300)
    with open(local_filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    extract_dir = os.path.join(download_dir, batch_id)
    with zipfile.ZipFile(local_filename, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def _get_batch_result(batch_id: str, download_dir: str) -> str:
    """轮询批量任务结果,完成后下载首个文件 zip 并解压,返回解压目录。"""
    url = f"{_MINERU_BASE}/extract-results/batch/{batch_id}"
    _WAITING = ("pending", "running", "converting", "waiting-file")
    max_retries = 120  # 最多轮询 120 × 5s = 10 分钟
    for retry in range(max_retries):
        res = requests.get(url, headers=_mineru_headers(), timeout=60)
        body = res.json()
        data = body.get("data", {})
        extract_result = data.get("extract_result") or []
        if not extract_result:
            state = data.get("state") or body.get("state") or "unknown"
            if data.get("err_msg"):
                raise RuntimeError(f"MinerU 任务出错: {data['err_msg']}")
            if state in _WAITING:
                time.sleep(5)
                continue
            raise RuntimeError(f"MinerU 未知响应: {body}")
        item = extract_result[0]
        state = item.get("state")
        if state in _WAITING:
            logger.info("MinerU 任务 %s,等待5秒... (%s/%s)", state, retry + 1, max_retries)
            time.sleep(5)
            continue
        if item.get("err_msg") or state == "failed":
            raise RuntimeError(f"MinerU 任务出错: {item.get('err_msg') or state}")
        if state == "done":
            full_zip_url = item.get("full_zip_url")
            if full_zip_url:
                return _download_and_unzip(full_zip_url, batch_id, download_dir)
            raise RuntimeError("MinerU 任务完成但未找到 full_zip_url")
        raise RuntimeError(f"MinerU 未知状态: {state}")
    raise RuntimeError(f"MinerU 轮询超时({max_retries * 5}s)")


def _extract_md(extract_dir: str, output_md_path: Path) -> Path:
    md_path = Path(extract_dir) / "full.md"
    if not md_path.exists():
        found = list(Path(extract_dir).rglob("*.md"))
        if not found:
            raise FileNotFoundError(f"未在 {extract_dir} 中找到 markdown 文件")
        md_path = found[0]
    output_md_path = Path(output_md_path)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(md_path), str(output_md_path))
    return output_md_path


def parse_pdf_mineru(local_path, output_md_path) -> Path:
    """MinerU 解析本地 PDF:申请上传链接 → PUT 上传 → 轮询 → 提取 full.md。"""
    local_path = Path(local_path)
    logger.info("MinerU 处理: %s", local_path.name)
    batch_id, file_urls = _apply_upload_url(local_path.name)
    logger.info("batch_id=%s, 上传中...", batch_id)
    _upload_file(local_path, file_urls[0])
    tmp_dir = tempfile.mkdtemp(prefix="mineru_")
    try:
        extract_dir = _get_batch_result(batch_id, download_dir=tmp_dir)
        return _extract_md(extract_dir, output_md_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
# pypdf 兜底方式
# ============================================================
def fallback_pdf_to_md(pdf_path, output_md_path) -> Path:
    """pypdf 按页抽取文本,写入 "# Page N" 标记,供 text_splitter 按页切分。

    扫描版 PDF 抽不出文本时,页文本为空(分块时自动过滤)。
    """
    from pypdf import PdfReader

    pdf_path = Path(pdf_path)
    output_md_path = Path(output_md_path)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(pdf_path))
    parts = []
    for i, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        parts.append(f"# Page {i}\n{text}")
    output_md_path.write_text("\n\n".join(parts), encoding="utf-8")
    logger.info("pypdf 兜底转换: %s → %s (%s 页)", pdf_path.name, output_md_path.name, len(reader.pages))
    return output_md_path


# ============================================================
# 批量转换入口
# ============================================================
def _pdf_stem(pdf_path: Path) -> str:
    return pdf_path.stem


def convert_all_pdfs(
    pdf_dir: Optional[str] = None,
    md_dir: Optional[str] = None,
    use_mineru: Optional[bool] = None,
    force_overwrite: bool = False,
) -> List[Path]:
    """批量转换 PDF → MD。

    :param pdf_dir: PDF 目录(默认 config.RAG_PDF_DIR)
    :param md_dir: MD 输出目录(默认 config.RAG_MD_DIR)
    :param use_mineru: True=MinerU;False=pypdf 兜底;None=有 key 用 MinerU,否则兜底
    :param force_overwrite: 已存在 .md 是否覆盖重转
    :return: 转换后 .md 路径列表
    """
    pdf_dir = Path(pdf_dir or config.RAG_PDF_DIR)
    md_dir = Path(md_dir or config.RAG_MD_DIR)
    md_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(p for p in pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("PDF 目录无 PDF 文件: %s", pdf_dir)
        return []

    if use_mineru is None:
        use_mineru = bool((getattr(config, "MINERU_API_KEY", "") or "").strip())

    outputs = []
    for pdf in pdfs:
        out_md = md_dir / f"{pdf.stem}.md"
        if out_md.exists() and not force_overwrite:
            logger.info("已存在,跳过: %s", out_md.name)
            outputs.append(out_md)
            continue
        try:
            if use_mineru:
                parse_pdf_mineru(pdf, out_md)
            else:
                fallback_pdf_to_md(pdf, out_md)
            outputs.append(out_md)
        except Exception as e:
            logger.error("转换失败 %s: %s(改用 pypdf 兜底)", pdf.name, e)
            try:
                fallback_pdf_to_md(pdf, out_md)
                outputs.append(out_md)
            except Exception as e2:
                logger.error("pypdf 兜底也失败 %s: %s", pdf.name, e2)
    return outputs


def ensure_md_files(
    pdf_dir: Optional[str] = None,
    md_dir: Optional[str] = None,
    force_overwrite: bool = False,
) -> bool:
    """确保 md_dir 有 .md 文件;无则按是否有 key 自动转换。返回是否用了 MinerU。

    日志关键点:明确告知是复用已有 MD 还是重新转换,以及是否用了 MinerU。
    若已有 MD 但想用 MinerU 重新转换,需传 force_overwrite=True 或先删除 md_dir。
    """
    pdf_dir = Path(pdf_dir or config.RAG_PDF_DIR)
    md_dir = Path(md_dir or config.RAG_MD_DIR)
    pdf_count = len(list(pdf_dir.glob("*.pdf"))) if pdf_dir.exists() else 0
    existing = list(md_dir.glob("*.md")) if md_dir.exists() else []
    has_key = bool((getattr(config, "MINERU_API_KEY", "") or "").strip())

    if existing and not force_overwrite:
        logger.info(
            "[MD状态] 已有 %s 个 MD(复用,跳过转换)| PDF=%s | MinerU key=%s",
            len(existing), pdf_count, "已配置" if has_key else "未配置",
        )
        if has_key:
            logger.info(
                "  已配置 MinerU key 但 MD 已存在故未转换;如需用 MinerU 重新转换,"
                "请删除 %s 或传 force_overwrite=True", md_dir,
            )
        return has_key

    logger.info(
        "[MD状态] %s | PDF=%s | 方式=%s",
        "重新转换(force_overwrite)" if force_overwrite else "无 MD,开始转换",
        pdf_count, "MinerU" if has_key else "pypdf兜底",
    )
    convert_all_pdfs(pdf_dir, md_dir, use_mineru=has_key, force_overwrite=force_overwrite)
    return has_key


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _force = "--convert" in sys.argv or "--force" in sys.argv
    used_mineru = ensure_md_files(force_overwrite=_force)
    print(f"转换完成,方式: {'MinerU' if used_mineru else 'pypdf兜底'}")
