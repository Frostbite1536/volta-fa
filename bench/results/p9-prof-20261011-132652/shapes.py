import sqlite3, sys, collections, re
db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("select id, value from StringIds"))
rows = db.execute("select demangledName, gridX, gridY, gridZ, blockX, end-start from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
agg = collections.defaultdict(lambda: [0, 0.0])
for nm, gx, gy, gz, bx, dur in rows:
    n = names[nm]
    if not re.search(r"mul_mat|mmq|quantize|prep", n): continue
    short = re.sub(r"\(const.*", "", n)[:70]
    agg[(short, gx, gy, gz, bx)][0] += 1; agg[(short, gx, gy, gz, bx)][1] += dur
for k, (c, t) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    print(f"{t/6e3:8.0f} us/eval {c/6:6.1f}x avg {t/c/1e3:7.1f} us grid {k[1]}x{k[2]}x{k[3]} blk {k[4]}  {k[0]}")
