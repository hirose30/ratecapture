#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  K近傍法による数字画像仕分けsample
#  =================================
#  Copyright (C) 2015 Takamasa Hirose
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
import copy
import random

import cv2
import numpy as np

# KNN samples
#knn = cv2.ml.KNearest_create()
knn = cv2.KNearest()
samples = None
responses = []


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--k',
        type=int,
        default=1,
        metavar='N',
        help='最近傍の図形の必要数 (num_samples <= N)'
    )
    return parser



def add_sample(response, img):
    global samples
    global responses

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sample = img.reshape((1, img.shape[0] * img.shape[1]))
    sample = np.array(sample, np.float32)

    if samples is None:
        samples = np.empty((0, img.shape[0] * img.shape[1]))
        responses = []

    samples = np.append(samples, sample, 0)
    responses.append(response)


def flame_sub(im1,im2,im3,th,blur):
    
    d1 = cv2.absdiff(im3, im2)
    d2 = cv2.absdiff(im2, im1)
    diff = cv2.bitwise_and(d1, d2)
    # 差分が閾値より小さければFalse
    return np.sum(th - diff) > diff.size
def main():
    # 引数のパース
    parser = create_parser()
    args = parser.parse_args()
    print(args)
    assert 0 < args.k
    k = args.k

    # ESCキーが押されるまでひたすら問題を解く
    keycode = 0
    
    cnt = 0
    tmp={}
    videofile = 'sample.mov'
    
    #learning
    cap = cv2.VideoCapture(videofile)
    pframe2 = cv2.cvtColor(cap.read()[1][4:-5,115:140], cv2.COLOR_RGB2GRAY)
    pframe1 = cv2.cvtColor(cap.read()[1][4:-5,115:140], cv2.COLOR_RGB2GRAY)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==False: 
            break
        copy.deepcopy(frame[4:-5,115:140])
        gray = cv2.cvtColor(copy.deepcopy(frame[4:-5,115:140]), cv2.COLOR_BGR2GRAY)
        if flame_sub(pframe2,pframe1,gray,1,1):
            cnt = 0
        cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame',48,70)
        cv2.imshow('frame',cv2.resize(gray, (48, 48)))
        
        pframe2 = pframe1
        pframe1 = gray
        cnt +=1
        if cnt == 2:
            var = raw_input("What's Number?:")
            add_sample(int(var),frame[4:-5,115:140])
            tmp[var] = tmp.get(var,0)+1
            wname = var+'_'+str(tmp[var])
            #cv2.namedWindow(wname,cv2.WINDOW_NORMAL)
            cv2.resizeWindow(wname,48,70)
            cv2.imshow(wname, cv2.resize(gray, (48, 48)))
            cv2.moveWindow(wname, 240+ 50 * (tmp[var]-1), 120 + 72 * (int(var) - 1))
        else:
            keycode = cv2.waitKey(1)
            
    cap.release()
    raw_input("Next?:")
    cv2.destroyAllWindows()
    
    print(u'(サンプル数, 次元数) = ', samples.shape)

    knn.train(
        np.array(samples, np.float32),
        #cv2.ml.ROW_SAMPLE,
        np.array(responses, np.float32),
    )

    #recognition
    cap = cv2.VideoCapture(videofile)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==False: 
            break
        img = copy.deepcopy(frame[4:-5,115:140])
        # K近傍法で最も近いサンプルを調べる
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sample = img2.reshape((1, img2.shape[0] * img2.shape[1]))
        sample = np.array(sample, np.float32)
        #retval, results, neigh_rep, dists = knn.findNearest(sample, k)
        retval, results, neigh_rep, dists = knn.find_nearest(sample, k)
        
        # 調べたサンプルのレスポンスから元の画像名を生成
        num_answer = int(results.ravel()[0])
        answer = str(num_answer)

        # 元の画像名のウインドウに表示
        cv2.imshow('source', cv2.resize(img, (48, 48)))
        cv2.moveWindow('source', 80, 20)
        cv2.imshow(answer, cv2.resize(img, (48, 48)))
        cv2.moveWindow(answer, 80, 240 + 64 * (num_answer - 1))
        keycode = cv2.waitKey(100)

main()