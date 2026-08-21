# -*- coding: utf-8 -*-
"""
数据集重新划分脚本
作用：从 train/ 随机移动 N 张图（+对应标注）到 valid/
目的：扩大 valid 集规模，让验证指标更稳定
注意：使用"移动"而非"复制"，避免同一样本同时出现在 train 和 valid（数据泄漏）

用法：
    python resplit_dataset.py                  # 默认 dry-run，只显示计划不执行
    python resplit_dataset.py --execute        # 真正执行移动
    python resplit_dataset.py --execute --target 150   # 指定 valid 目标数量
    python resplit_dataset.py --rollback       # 回滚（把 valid 的图移回 train）
"""
import argparse
import random
import shutil
from pathlib import Path

# ============ 配置 ============
DATA_ROOT = Path(__file__).resolve().parent / "MaskDataSet"
TRAIN_IMG_DIR = DATA_ROOT / "train" / "images"
TRAIN_LBL_DIR = DATA_ROOT / "train" / "labels"
VALID_IMG_DIR = DATA_ROOT / "valid" / "images"
VALID_LBL_DIR = DATA_ROOT / "valid" / "labels"
TEST_IMG_DIR = DATA_ROOT / "test" / "images"
TEST_LBL_DIR = DATA_ROOT / "test" / "labels"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SEED = 42  # 固定随机种子，保证可重现


def count_images(d: Path) -> int:
    """统计目录下图片数量"""
    if not d.exists():
        return 0
    return sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def find_label_for_image(img_path: Path, lbl_dir: Path) -> Path | None:
    """根据图片路径找对应标注文件（同名 .txt）"""
    lbl = lbl_dir / (img_path.stem + ".txt")
    return lbl if lbl.exists() else None


def move_sample(img: Path, lbl: Path | None, src_img_dir: Path, src_lbl_dir: Path,
                dst_img_dir: Path, dst_lbl_dir: Path) -> tuple[Path, Path | None]:
    """移动一个样本（图片+标注）到目标目录，返回新路径"""
    new_img = dst_img_dir / img.name
    new_lbl = None
    shutil.move(str(img), str(new_img))
    if lbl is not None:
        new_lbl = dst_lbl_dir / lbl.name
        shutil.move(str(lbl), str(new_lbl))
    return new_img, new_lbl


def clear_labels_cache():
    """删除 ultralytics 的 labels.cache 文件，让下次训练时重新扫描"""
    deleted = []
    for split_dir in [TRAIN_LBL_DIR, VALID_LBL_DIR, TEST_LBL_DIR]:
        cache = split_dir / "labels.cache"
        if cache.exists():
            cache.unlink()
            deleted.append(str(cache))
    return deleted


def show_stats():
    """打印当前各集合统计"""
    train_n = count_images(TRAIN_IMG_DIR)
    valid_n = count_images(VALID_IMG_DIR)
    test_n = count_images(TEST_IMG_DIR)
    total = train_n + valid_n + test_n
    print("\n========== 数据集统计 ==========")
    print(f"train: {train_n:4d} 张  ({train_n/total*100:.1f}%)")
    print(f"valid: {valid_n:4d} 张  ({valid_n/total*100:.1f}%)")
    print(f"test : {test_n:4d} 张  ({test_n/total*100:.1f}%)")
    print(f"total: {total:4d} 张")
    return train_n, valid_n, test_n


def plan_move(target_valid: int) -> list[tuple[Path, Path | None]]:
    """规划要从 train 移动到 valid 的样本列表
    返回 [(img_path, lbl_path), ...]
    """
    train_imgs = sorted([f for f in TRAIN_IMG_DIR.iterdir() if f.suffix.lower() in IMAGE_EXTS])
    valid_n = count_images(VALID_IMG_DIR)
    need = target_valid - valid_n
    if need <= 0:
        print(f"[INFO] valid 已有 {valid_n} 张，目标 {target_valid} 张，无需移动")
        return []
    if need > len(train_imgs):
        print(f"[ERROR] train 只有 {len(train_imgs)} 张，无法移出 {need} 张到 valid")
        return []
    random.seed(SEED)
    selected = random.sample(train_imgs, need)
    plan = []
    for img in selected:
        lbl = find_label_for_image(img, TRAIN_LBL_DIR)
        plan.append((img, lbl))
    return plan


def do_execute(target_valid: int):
    """执行移动：train → valid"""
    plan = plan_move(target_valid)
    if not plan:
        return
    print(f"\n[计划] 将从 train 移动 {len(plan)} 张到 valid")
    print(f"[计划] random.seed={SEED}（可重现）")
    print(f"\n前 5 个将移动的样本：")
    for img, lbl in plan[:5]:
        print(f"  {img.name}  +  {lbl.name if lbl else '(无标注!)'}")
    if len(plan) > 5:
        print(f"  ... 共 {len(plan)} 个")

    # 检查是否有缺失标注的样本
    missing_lbl = [img for img, lbl in plan if lbl is None]
    if missing_lbl:
        print(f"\n[WARN] {len(missing_lbl)} 个样本缺失标注文件，将只移动图片：")
        for img in missing_lbl[:3]:
            print(f"  {img.name}")
        if len(missing_lbl) > 3:
            print(f"  ... 共 {len(missing_lbl)} 个")

    print("\n开始执行移动...")
    moved = 0
    for i, (img, lbl) in enumerate(plan, 1):
        try:
            move_sample(img, lbl, TRAIN_IMG_DIR, TRAIN_LBL_DIR,
                        VALID_IMG_DIR, VALID_LBL_DIR)
            moved += 1
            if i % 20 == 0 or i == len(plan):
                print(f"  进度: {i}/{len(plan)}")
        except Exception as e:
            print(f"[ERROR] 移动 {img.name} 失败: {e}")

    # 清理 cache
    deleted = clear_labels_cache()
    if deleted:
        print(f"\n[INFO] 已删除 labels.cache（让 ultralytics 重新生成）:")
        for c in deleted:
            print(f"  {c}")

    print(f"\n[完成] 成功移动 {moved}/{len(plan)} 个样本")
    show_stats()
    print(f"\n[下一步] 重新训练: python yolov8_mask_v2\\yolov8_mask.py")


def do_rollback():
    """回滚：把 valid 全部移回 train"""
    valid_imgs = sorted([f for f in VALID_IMG_DIR.iterdir() if f.suffix.lower() in IMAGE_EXTS])
    if not valid_imgs:
        print("[INFO] valid 目录为空，无需回滚")
        return
    print(f"\n[回滚] 将把 valid 全部 {len(valid_imgs)} 张移回 train")
    moved = 0
    for img in valid_imgs:
        try:
            lbl = find_label_for_image(img, VALID_LBL_DIR)
            move_sample(img, lbl, VALID_IMG_DIR, VALID_LBL_DIR,
                        TRAIN_IMG_DIR, TRAIN_LBL_DIR)
            moved += 1
        except Exception as e:
            print(f"[ERROR] 回滚 {img.name} 失败: {e}")
    clear_labels_cache()
    print(f"[完成] 回滚 {moved} 个样本")
    show_stats()


def main():
    parser = argparse.ArgumentParser(description="数据集重新划分：train → valid")
    parser.add_argument("--execute", action="store_true",
                        help="真正执行移动（默认只 dry-run）")
    parser.add_argument("--target", type=int, default=150,
                        help="valid 目标数量（默认 150）")
    parser.add_argument("--rollback", action="store_true",
                        help="回滚：把 valid 全部移回 train")
    args = parser.parse_args()

    print(f"[配置] DATA_ROOT = {DATA_ROOT}")
    print(f"[配置] random.seed = {SEED}")

    show_stats()

    if args.rollback:
        do_rollback()
        return

    if not args.execute:
        # dry-run 模式：只显示计划
        plan = plan_move(args.target)
        if not plan:
            return
        print(f"\n[DRY-RUN] 计划从 train 移动 {len(plan)} 张到 valid（target={args.target}）")
        print(f"[DRY-RUN] 前 5 个将移动的样本：")
        for img, lbl in plan[:5]:
            print(f"  {img.name}  +  {lbl.name if lbl else '(无标注!)'}")
        if len(plan) > 5:
            print(f"  ... 共 {len(plan)} 个")
        print(f"\n[提示] 这只是预览，未实际移动。确认后执行：")
        print(f"  python resplit_dataset.py --execute --target {args.target}")
        print(f"\n[回滚] 如果想撤销，执行：")
        print(f"  python resplit_dataset.py --rollback")
        return

    # 真正执行
    do_execute(args.target)


if __name__ == "__main__":
    main()
