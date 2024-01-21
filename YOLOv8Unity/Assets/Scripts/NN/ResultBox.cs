using Unity.Barracuda;
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

        public static ResultBox DecodeBox(float[,] array, int box, DecodeParams decodeParams)
        {
            (int highestClassIndex, float highestScore) = DecodeBestBoxIndexAndScore(array, box, decodeParams);

            if (highestScore < decodeParams.DiscardThreshold)
                return null;

            Rect box_rect = DecodeBoxRectangle(array, box, decodeParams);

            ResultBox result = new(
                rect: box_rect,
                score: highestScore,
                bestClassIndex: highestClassIndex);
            return result;
        }

        private static (int, float) DecodeBestBoxIndexAndScore(float[,] array, int box, DecodeParams decodeParams)
        {
            const int classesOffset = 4;

            int highestClassIndex = 0;
            float highestScore = 0;

            for (int i = 0; i < decodeParams.ClassesNum; i++)
            {
                float currentClassScore = array[box, i + classesOffset];
                if (currentClassScore > highestScore)
                {
                    highestScore = currentClassScore;
                    highestClassIndex = i;
                }
            }

            return (highestClassIndex, highestScore);
        }

        private static Rect DecodeBoxRectangle(float[,] data, int box, DecodeParams decodeParams)
        {
            const int boxCenterXIndex = 0;
            const int boxCenterYIndex = 1;
            const int boxWidthIndex = 2;
            const int boxHeightIndex = 3;

            float centerX = data[box, boxCenterXIndex];
            float centerY = data[box, boxCenterYIndex];
            float width = data[box, boxWidthIndex];
            float height = data[box, boxHeightIndex];

            float xMin = centerX - width / 2;
            float yMin = centerY - height / 2;
            xMin = xMin < 0 ? 0 : xMin;
            yMin = yMin < 0 ? 0 : yMin;
            var rect = new Rect(xMin, yMin, width, height);
            rect.xMax = rect.xMax > decodeParams.InputWidth ? decodeParams.InputWidth : rect.xMax;
            rect.yMax = rect.yMax > decodeParams.InputHeight ? decodeParams.InputHeight : rect.yMax;

            return rect;
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
            maskInd?.tensorOnDevice.Dispose();
        }
    }

    public class ResultBoxWithMasks : ResultBox
    {
        public Tensor masks;
        public readonly Tensor maskInd;

        public ResultBoxWithMasks(ResultBox box, Tensor masks, Tensor masksScores) : base(box.rect, box.score, box.bestClassIndex)
        {
            this.masks = masks;
            maskInd = masksScores;
        }

        ~ResultBoxWithMasks()
        {
            maskInd?.tensorOnDevice.Dispose();
            masks?.tensorOnDevice.Dispose();
        }
    }
}