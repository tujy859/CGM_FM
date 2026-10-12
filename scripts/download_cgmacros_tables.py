"""高效下载 CGMacros 数据集中的所有表格文件（CSV/元数据）。
利用 HTTP Range 请求从 PhysioNet 的 626MB 压缩包中仅提取 6.3MB 的 CSV 数据，
跳过不需要的图像文件，实现极速下载。
产物存放于 datasets/CGMacros/。
"""
import io
import os
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = ROOT / "datasets" / "CGMacros"
URL = "https://physionet.org/files/cgmacros/1.0.0/CGMacros_dateshifted365.zip"


def fetch_range(url: str, start: int, end: int, retries: int = 5, delay: float = 2.0) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                if resp.status not in (200, 206):
                    raise RuntimeError(f"Unexpected status: {resp.status}")
                return resp.read()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("CGMacros 表格数据定向下载脚本")
    print(f"目标目录: {TARGET_DIR}")
    print(f"远程源: {URL}")
    print("=" * 60)

    # 1. 获取 ZIP 尾部，解析 Central Directory
    print("\n[1/3] 读取 ZIP Central Directory 索引...")
    # 请求末尾 512KB
    req_tail = urllib.request.Request(URL, headers={"Range": "bytes=-524288"})
    with urllib.request.urlopen(req_tail, timeout=20) as resp:
        tail = resp.read()
        content_range = resp.headers.get("Content-Range")
        total_len = int(content_range.split("/")[-1])

    tail_start = total_len - len(tail)
    eocd_idx = tail.rfind(b"PK\x05\x06")
    if eocd_idx == -1:
        raise RuntimeError("未在末尾找到 EOCD 标识")

    eocd = tail[eocd_idx : eocd_idx + 22]
    _, _, _, num_entries, cd_size, cd_offset, _ = struct.unpack("<HHHHIIH", eocd[4:22])
    print(f"  总大小: {total_len / 1024 / 1024:.1f} MB, 总条目数: {num_entries}")

    if cd_offset >= tail_start:
        cd_bytes = tail[cd_offset - tail_start : cd_offset - tail_start + cd_size]
    else:
        print("  Central Directory 超出尾部缓冲，单独拉取...")
        cd_bytes = fetch_range(URL, cd_offset, cd_offset + cd_size - 1)

    # 2. 遍历索引筛选所有 CSV 文件
    p = 0
    csv_items = []
    while p < len(cd_bytes):
        if cd_bytes[p : p + 4] != b"PK\x01\x02":
            break
        (
            _,
            _,
            _,
            method,
            _,
            _,
            _,
            comp_size,
            uncomp_size,
            name_len,
            extra_len,
            comm_len,
            _,
            _,
            _,
            local_offset,
        ) = struct.unpack("<HHHHHHIIIHHHHHII", cd_bytes[p + 4 : p + 46])
        name = cd_bytes[p + 46 : p + 46 + name_len].decode("utf-8", errors="ignore")
        if name.endswith(".csv"):
            csv_items.append(
                {
                    "name": name,
                    "method": method,
                    "comp_size": comp_size,
                    "uncomp_size": uncomp_size,
                    "local_offset": local_offset,
                }
            )
        p += 46 + name_len + extra_len + comm_len

    print(f"\n[2/3] 检索到 CSV 数据表: {len(csv_items)} 个")
    total_comp_mb = sum(x["comp_size"] for x in csv_items) / 1024 / 1024
    total_uncomp_mb = sum(x["uncomp_size"] for x in csv_items) / 1024 / 1024
    print(f"  压缩体积: {total_comp_mb:.2f} MB, 解压体积: {total_uncomp_mb:.2f} MB")

    # 3. 逐个流式下载并解压写入
    print("\n[3/3] 开始定向下载并解压...")
    downloaded = 0
    for idx, item in enumerate(csv_items, 1):
        rel_path = item["name"]
        # 去掉第一层 CGMacros/ 前缀，避免多重嵌套
        parts = Path(rel_path).parts
        if parts[0] == "CGMacros":
            out_file = TARGET_DIR.joinpath(*parts[1:])
        else:
            out_file = TARGET_DIR / rel_path

        out_file.parent.mkdir(parents=True, exist_ok=True)

        if out_file.exists() and out_file.stat().st_size == item["uncomp_size"]:
            downloaded += 1
            print(f"  [{idx:02d}/{len(csv_items)}] 已存在 (跳过): {out_file.name}")
            continue

        loc = item["local_offset"]
        # 获取 local header
        hdr_data = fetch_range(URL, loc, loc + 300)
        _, _, _, _, _, _, _, _, _, l_name_len, l_extra_len = struct.unpack(
            "<IHHHHHIIIHH", hdr_data[:30]
        )
        data_start = loc + 30 + l_name_len + l_extra_len
        data_end = data_start + item["comp_size"] - 1

        comp_data = fetch_range(URL, data_start, data_end)
        if item["method"] == 8:
            raw_data = zlib.decompress(comp_data, -15)
        elif item["method"] == 0:
            raw_data = comp_data
        else:
            raise ValueError(f"不支持的压缩方式: {item['method']}")

        with open(out_file, "wb") as f:
            f.write(raw_data)

        downloaded += 1
        print(
            f"  [{idx:02d}/{len(csv_items)}] 下载解压完成: {out_file.name} ({len(raw_data)/1024:.1f} KB)"
        )

    print("\n" + "=" * 60)
    print(f"全部 {downloaded} 个 CSV 数据表已成功落地至: {TARGET_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
