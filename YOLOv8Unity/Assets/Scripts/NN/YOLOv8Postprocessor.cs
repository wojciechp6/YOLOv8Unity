using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Profiling;

namespace NN
{
    static public class YOLOv8Postprocessor
    {
        public static float DiscardThreshold = 0.1f;
        const int ClassesNum = 80;
        const int BoxesPerCell = 8400;
        const int WidthHeight = 640;

        static public List<ResultBox> DecodeNNOut(Tensor output)
        {
            float[,] array = ReadOutputToArray(output);
            Profiler.BeginSample("YOLOv8Postprocessor.DecodeNNOut");
            List<ResultBox> boxes = DecodeCell(array).ToList();
            Profiler.EndSample();
            return boxes;
        }

        private static float[,] ReadOutputToArray(Tensor output)
        {
            var reshapedOutput = output.Reshape(new[] { 1, 1, BoxesPerCell, -1 });
            var array = TensorToArray2D(reshapedOutput);
            reshapedOutput.Dispose();
            return array;
        }

        private static IEnumerable<ResultBox> DecodeCell(float[,] array)
        {
            int boxes = array.GetLength(0);
            for (int box_index = 0; box_index < boxes; box_index++)
            {
                var box = DecodeBox(array, box_index);
                if (box != null)
                    yield return box;
            }
        }

        static private ResultBox DecodeBox(float[,] array, int box)
        {
            (int highestClassIndex, float highestScore) = DecodeBestBoxIndexAndScore(array, box);

            if (highestScore < DiscardThreshold)
                return null;

            Rect box_rect = DecodeBoxRectangle(array, box);

            int masksNum = array.GetLength(1) - 4 - ClassesNum;
            float[] masksScore = Array2DTo1DCopy(array, box, 4 + ClassesNum, 32);


            var result = new ResultBox
            {
                rect = box_rect,
                confidence = highestScore,
                bestClassIdx = highestClassIndex,
                maskInd = new(1, masksScore.Length, masksScore),
            };
            return result;
        }

        static private (int, float) DecodeBestBoxIndexAndScore(float[,] array, int box)
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

        static private Rect DecodeBoxRectangle(float[,] data, int box)
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
            rect.xMax = rect.xMax > WidthHeight ? WidthHeight : rect.xMax;
            rect.yMax = rect.yMax > WidthHeight ? WidthHeight : rect.yMax;

            return rect;
        }

        private static float[,] TensorToArray2D(this Tensor tensor)
        {
            float[,] output = new float[tensor.width, tensor.channels];
            var data = tensor.AsFloats();
            int bytes = Buffer.ByteLength(data);
            Buffer.BlockCopy(data, 0, output, 0, bytes);
            return output;
        }

        private static T[] Array2DTo1DCopy<T>(T[,] inputArray, int firstDimmension, int secondDimmension, int count)
        {
            int tSize = Marshal.SizeOf<T>();
            int start = firstDimmension * inputArray.GetLength(1) + secondDimmension;
            T[] output = new T[count];
            Buffer.BlockCopy(inputArray, start * tSize, output, 0, count * tSize);
            return output;
        }
    }
}