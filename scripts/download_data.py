"""下载 MovieLens 100K 的 u.data 到项目 data/ml-100k/ 目录（仅用标准库）。

源 1：GroupLens 官方；源 2：GitHub raw 镜像（部分网络环境下官方源不可达）。
若全部失败，手动下载 u.data 放入 data/ml-100k/ 即可，本环境只用这一个文件。
"""
import os
import urllib.request

URLS = [
    "https://files.grouplens.org/datasets/movielens/ml-100k/u.data",
    "https://raw.githubusercontent.com/vinjn/MLStudy/master/data/movielens-100k/u.data",
]


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target = os.path.join(root, "data", "ml-100k")
    if os.path.exists(os.path.join(target, "u.data")):
        print("数据已存在:", target)
        return
    os.makedirs(target, exist_ok=True)
    last_err = None
    for url in URLS:
        try:
            print("下载:", url)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, \
                    open(os.path.join(target, "u.data"), "wb") as f:
                f.write(resp.read())
            print("完成，数据文件:", os.path.join(target, "u.data"))
            return
        except Exception as e:  # 当前源不可达，尝试下一个
            last_err = e
            print("该源失败:", e)
    raise SystemExit(f"所有源均失败，请手动下载 u.data 放到 {target}/ 目录: {last_err}")


if __name__ == "__main__":
    main()
