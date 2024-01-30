﻿using Unity.Barracuda;
using UnityEngine;

namespace NN
{
    public readonly struct DecodeParams
    {
        public readonly float DiscardThreshold;
        public readonly int ClassesNum;
        public readonly int InputWidth;
        public readonly int InputHeight;

        public DecodeParams(float discardThreshold, int classesNum, int inputWidth, int inputHeight)
        {
            DiscardThreshold = discardThreshold;
            ClassesNum = classesNum;
            InputWidth = inputWidth;
            InputHeight = inputHeight;
        }
    }
    public class ResultBox
    {
        public readonly Rect rect;
        public float score;
        public readonly int bestClassIndex;

        public ResultBox(Rect rect, float score, int bestClassIndex)
        {
            this.rect = rect;
            this.score = score;
            this.bestClassIndex = bestClassIndex;
        }
    }

    public class ResultBoxWithMasksIndices : ResultBox
    {
        public readonly Tensor maskInd;

        public ResultBoxWithMasksIndices(ResultBox box, Tensor maskInd) : base(box.rect, box.score, box.bestClassIndex)
        {
            this.maskInd = maskInd;
        }

        ~ResultBoxWithMasksIndices()
        {
            maskInd.tensorOnDevice.Dispose();
        }
    }

    public class ResultBoxWithMask : ResultBox
    {
        public readonly Tensor masks;

        public ResultBoxWithMask(ResultBox box, Tensor masks) : base(box.rect, box.score, box.bestClassIndex)
        {
            this.masks = masks;
        }

        ~ResultBoxWithMask()
        {
            masks.tensorOnDevice.Dispose();
        }
    }
}