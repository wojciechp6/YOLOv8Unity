using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Profiling;

namespace NN
{
    public class YOLOv8OutputReader
    {
        public static float DiscardThreshold = 0.1f;
        protected const int ClassesNum = 80;
        const int BoxesPerCell = 8400;
        const int InputWidth = 640;
        const int InputHeight = 640;

        public IEnumerable<ResultBox> ReadOutput(Tensor output)
        {
            float[,] array = ReadOutputToArray(output);
            foreach (ResultBox result in ReadBoxes(array))
                yield return result;
        }


        private float[,] ReadOutputToArray(Tensor output)
        {
            var reshapedOutput = output.Reshape(new[] { 1, 1, BoxesPerCell, -1 });
            var array = TensorToArray2D(reshapedOutput);
            reshapedOutput.Dispose();
            return array;
        }

        private IEnumerable<ResultBox> ReadBoxes(float[,] array)
        {
            int boxes = array.GetLength(0);
            for (int box_index = 0; box_index < boxes; box_index++)
            {
                ResultBox box = ReadBox(array, box_index);
                if (box != null)
                    yield return box;
            }
        }

        protected virtual ResultBox ReadBox(float[,] array, int box)
        {
            (int highestClassIndex, float highestScore) = DecodeBestBoxIndexAndScore(array, box);

            if (highestScore < DiscardThreshold)
                return null;

            Rect box_rect = DecodeBoxRectangle(array, box);

            ResultBox result = new(
                rect: box_rect,
                score: highestScore,
                bestClassIndex: highestClassIndex);
            return result;
        }

        private (int, float) DecodeBestBoxIndexAndScore(float[,] array, int box)
        {
            const int classesOffset = 4;

            int highestClassIndex = 0;
            float highestScore = 0;

            for (int i = 0; i < ClassesNum; i++)
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

        private Rect DecodeBoxRectangle(float[,] data, int box)
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
            rect.xMax = rect.xMax > InputWidth ? InputWidth : rect.xMax;
            rect.yMax = rect.yMax > InputHeight ? InputHeight : rect.yMax;

            return rect;
        }

        private float[,] TensorToArray2D(Tensor tensor)
        {
            float[,] output = new float[tensor.width, tensor.channels];
            var data = tensor.AsFloats();
            int bytes = Buffer.ByteLength(data);
            Buffer.BlockCopy(data, 0, output, 0, bytes);
            return output;
        }
    }
}