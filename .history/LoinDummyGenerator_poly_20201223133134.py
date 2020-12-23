#%%
import cv2
import numpy as np
import math
from numpy.random import *
from tqdm import tqdm 

# 画像サイズ指定
height=300
width=300

for i in tqdm(range(10)):
    # 脂肪（白）領域
    # 中心を中心x,y±10%の範囲でランダムに決定。
    outcenterx = width/2 + rand()*width/10 - width/20
    outcentery = height/2 + rand()*height/10 - height/20
    # 領域のv,hサイズを200～250の範囲でランダムに決定。
    outrangev = rand()*50 + 230
    outrangeh = rand()*50 + 230
    # 領域を±90度の範囲で回転。
    outangle = rand()*180-90
    # 領域の頂点の数
    outpoints = 3 + int(rand()*8)

    # ロース芯を２つの楕円で表現
    # ロース芯１の中心座標xを脂肪領域中心－0～50で決定。
    loin1centerx = outcenterx - rand()*50
    # ロース芯１の中心座標yを脂肪領域中心±25で決定。
    loin1centery = outcentery + rand()*50 - 25
    # ロース芯１の領域hを脂肪領域外縁に接する円x0.8～1.0倍に決定。外縁に触れるか触れないかの大きさに。
    loin1rangeh = (outrangeh/2 - (outcenterx - loin1centerx))*2*(0.8 + rand()*0.2)
    # ロース芯１の領域vをhの0.8～1.2倍に決定。
    loin1rangev = loin1rangeh * (0.8 + rand()*0.4)
    # ロース芯１を±90度の範囲で回転。
    loin1angle = rand()*180-90
    # ロース芯１の頂点の数
    loin1points = 4 + int(rand()*8)

    # ロース芯２の中心座標xを脂肪領域中心＋0～50で決定。
    loin2centerx = outcenterx + rand()*50
    # ロース芯２の中心座標yを脂肪領域中 心±25で決定。
    loin2centery = outcentery + rand()*50 - 25
    # ロース芯２の領域hを脂肪領域外縁に接する円x0.8～1.0倍に決定。外縁に触れるか触れないかの大きさに。
    loin2rangeh = (outrangeh/2 - (loin2centerx - outcenterx))*2*(0.8 + rand()*0.2)
    # ロース芯２の領域vをhの0.8～1.2倍に決定。
    loin2rangev = loin2rangeh * (0.8 + rand()*0.4)
    # ロース芯２を±90度の範囲で回転。
    loin2angle = rand()*180-90 
    # ロース芯２の頂点の数
    loin2points = 4 + int(rand()*8)

    # 疑似脂肪交雑レート
    marblerate = rand()*0.5
 
    # 多角形描画
    # 白紙の背景を作製
    img = np.full((height, width, 1), np.float(255), dtype=np.uint8)

    # 黒塗りの脂肪領域(多角形）を作製
    outpts = []
    outbaseangle = int(360/outpoints)
    angle = 0 - rand()*outbaseangle*0.2
    while angle < 360:
        angle = angle + outbaseangle
        rad = math.radians(angle)
        r = (outrangev + rand()*(outrangeh - outrangev))/2
        outpts.append([outcenterx + r*np.cos(rad), outcentery + r*np.sin(rad)])
        angle = angle - rand()*outbaseangle*0.2
    pts = np.array(outpts).reshape((-1,1,2)).astype(np.int32)
    img = cv2.fillPoly(img, [pts] , color = 0)

    # 白塗りのロース芯画像多角形１を作製
    loin1pts = []
    loin1baseangle = int(360/loin1points)
    angle = 0 - rand()*loin1baseangle*0.2
    while angle < 360:
        angle = angle + loin1baseangle
        rad = math.radians(angle)
        r = (loin1rangev + rand()*(loin1rangeh - loin1rangev))/2
        loin1pts.append([loin1centerx + r*np.cos(rad), loin1centery + r*np.sin(rad)])
        angle = angle - rand()*loin1baseangle*0.2
    pts1 = np.array(loin1pts).reshape((-1,1,2)).astype(np.int32)
    img = cv2.fillPoly(img, [pts1] , color = 255)

    # 白塗りのロース芯画像多角形２を作製
    loin2pts = []
    loin2baseangle = int(360/loin2points)
    angle = 0 - rand()*loin2baseangle*0.2
    while angle < 360:
        angle = angle + loin2baseangle
        rad = math.radians(angle)
        r = (loin2rangev + rand()*(loin2rangeh - loin2rangev))/2
        loin2pts.append([loin2centerx + r*np.cos(rad), loin2centery + r*np.sin(rad)])
        angle = angle - rand()*loin2baseangle*0.2
    pts2 = np.array(loin2pts).reshape((-1,1,2)).astype(np.int32)
    img = cv2.fillPoly(img, [pts2] , color = 255)

    # 脂肪交雑ノイズを入れ、白黒反転
    marble = rand(width, height, 1)
    marble = np.where(marble <= marblerate, np.float(0), np.float(1))
    img = img * marble
    img = 255 - img

    # 答えとなるロース芯領域画像の作成
    bwimg = np.full((height, width, 1), np.float(0), dtype=np.uint8)
    bwimg = cv2.fillPoly(bwimg, [pts1] , color = 255)
    bwimg = cv2.fillPoly(bwimg, [pts2] , color = 255)

    # ファイル書き込み
    cv2.imwrite('./img/demo_'+str(i)+'.bmp', img)
    cv2.imwrite('./bwimg/mask_'+str(i)+'.bmp', bwimg)



# %%
