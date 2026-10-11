import sqlite3, sys, statistics
db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("select id, value from StringIds"))
rows = db.execute("select start, end, gridX, gridY, blockX, shortName, demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
sel = [(s, e, i) for i, (s, e, gx, gy, bx, sn, dn) in enumerate(rows) if gx == 1 and gy in (10, 20) and 'mul_mat_q_volta' in names.get(dn, '')]
d = [(e - s)/1000 for s, e, i in sel]
print('n', len(d), 'min %.1f p10 %.1f med %.1f p90 %.1f max %.1f' % (min(d), sorted(d)[len(d)//10], statistics.median(d), sorted(d)[9*len(d)//10], max(d)))
# what runs right before/after a few of them
for s, e, i in sel[:3]:
    for j in range(i-2, i+3):
        S, E, gx, gy, bx, sn, dn = rows[j]
        print('   %8.1f us  gap %6.1f  %s grid %dx%d' % ((E-S)/1000, (S - rows[j-1][1])/1000, names.get(dn, '')[:70], gx, gy))
    print()
