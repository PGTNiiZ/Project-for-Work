# Error Analysis — Production Decision Matrix

## 0) คู่จริงที่ blocking พลาด (ไม่เคยถูก score) — 3,316 คู่
ส่วนที่หายของ FN 5,554 ในรูป 4.14 (= 2,242 โมเดลปัดตก + 3,316 ไม่เข้า candidate)

name similarity (max ของ 4 ช่องชื่อ):
name_sim_max
<0.3       1140
0.3-0.5     993
0.5-0.7     565
0.7-0.9     472
>=0.9       146

platform pair:
googleplus x twitter       1659
googleplus x instagram     1048
instagram x twitter         605
googleplus x googleplus       4

field ครบทั้งคู่ (%): {"bio_both_present_pct": 57.1, "location_both_present_pct": 0.0, "externalUrl_both_present_pct": 55.8}

## 1) False Positive — MATCH แต่ไม่ใช่คนเดียวกัน (925 คู่)

decision_source:
decision_source
AUTO_HIGH     862
AUTO_EXACT     63

userName เหมือนกันเป๊ะ: 10 คู่ (1.1%)

name similarity:
name_sim_max
<0.3        30
0.3-0.5     88
0.5-0.7    653
0.7-0.9     51
>=0.9      103

platform pair:
googleplus x twitter      431
googleplus x instagram    252
instagram x twitter       242

ตัวอย่าง exact-FP (username ชนกันแต่คนละคน):

- `weareccad` (instagram, folder=nicolacraddock) vs `ccad-clevelandcollegeofartdesign` (googleplus, folder=ccad)
- `weareccad` (instagram, folder=nicolacraddock) vs `weareccad` (twitter, folder=ccad)
- `wallhattori` (googleplus, folder=superwall13) vs `wallhattori` (instagram, folder=WallHattori)
- `wallhattori` (googleplus, folder=superwall13) vs `wallhattori` (twitter, folder=WallHattori)
- `tonybarlow` (instagram, folder=tonybarlow) vs `tonybarlow` (googleplus, folder=saltbar)
- `pullbackes` (instagram, folder=PullbackES) vs `pullbackes` (twitter, folder=espullback)
- `deakaipo` (twitter, folder=DeaKaipo) vs `miragusa` (instagram, folder=MichelleRagusa)
- `guiamonica` (twitter, folder=guiamonica) vs `foodreviewsmanila` (instagram, folder=FoodReviewsManila)

## 2) False Negative (scored) — NO_MATCH แต่คือคนเดียวกัน (2,242 คู่)

score distribution:
min    0.0041
25%    0.6061
50%    0.8316
75%    0.9286
max    0.9440

score >= 0.5 (เกือบถึง review band 0.95): 1,830 คู่
score >= 0.90: 765 คู่

name similarity:
name_sim_max
<0.3         16
0.3-0.5     230
0.5-0.7     531
0.7-0.9    1013
>=0.9       452

platform pair:
googleplus x twitter      993
googleplus x instagram    780
instagram x twitter       469

field ครบทั้งคู่ (%): {"bio_both_present_pct": 56.4, "location_both_present_pct": 0.0, "externalUrl_both_present_pct": 59.9}

## 3) REVIEW queue — 86,296 คู่ (จริง 4,065 / ไม่จริง 82,231 — precision 0.047)

precision ตาม score band ใน REVIEW:
                   n  true  precision
band                                 
(0.95, 0.955]    178     1     0.0056
(0.955, 0.96]  57080   724     0.0127
(0.96, 0.965]   6031   146     0.0242
(0.965, 0.97]     60     1     0.0167
(0.97, 0.975]     59     0     0.0000
(0.975, 0.98]  22888  3193     0.1395

precision ตาม name similarity ใน REVIEW:
             n  true  precision
ns_band                        
<0.3        59     0     0.0000
0.3-0.5   6283    53     0.0084
0.5-0.7  72461   674     0.0093
0.7-0.9   4622   750     0.1623
>=0.9     2871  2588     0.9014

platform pair (เฉพาะคู่จริงใน REVIEW):
googleplus x twitter      1808
googleplus x instagram    1776
instagram x twitter        481
