# # $net = 1, 2, 3, 5, 9, 8, 12, 15, 17
# # $net = 1, 6, 13, 3, 4, 5, 7, 2, 17
# $net = 1, 16, 18, 6, 7, 2, 4, 3, 8
# $flow = 1, 2, 3, 4, 5
# for ($i = 0; $i -le ($net.length - 1); $i += 1) {
#     for ($j = 0; $j -le ($flow.length - 1); $j += 1) {
#         python ./optimization.py input/network/feed_forward/random/4/net$($net[$i]).npy input/flow/random/4/flow$($flow[$j]).npz output --two-slope
#     }
# }

for ($j = 1; $j -le 1000; $j += 1) {
    python ./optimization.py input/network/feed_forward/parking_lot/6/net1.npy input/flow/parking_lot/6/flow$($j).npz output/parking_lot --two-slope --fast 2
}
