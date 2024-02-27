using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Profiling;

namespace NN
{
    public class YOLOv8SegmentationOutputReader : YOLOv8OutputReader
    {
        protected override ResultBox ReadBox(float[,] array, int box)
        {
            ResultBox resultBox = base.ReadBox(array, box);
            if (resultBox == null)
                return null;

            float[] masksScore = Array2DTo1DCopy(array, box, 4 + ClassesNum, 32);
            Tensor masksScoresTensor = new(1, masksScore.Length, masksScore);

            ResultBoxWithMasksIndices result = new(resultBox, masksScoresTensor);
            return result;
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