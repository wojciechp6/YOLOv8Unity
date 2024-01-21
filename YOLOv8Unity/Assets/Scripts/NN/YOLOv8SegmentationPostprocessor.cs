using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Profiling;

namespace NN
{
    public class YOLOv8SegmentationPostprocessor : YOLOv8Postprocessor<ResultBoxWithMasks>
    {
        public static float DiscardThreshold = 0.1f;
        const int ClassesNum = 80;
        const int BoxesPerCell = 8400;
        const int WidthHeight = 640;
        

        protected override List<ResultBoxWithMasks> DecodeNNOut(Tensor[] outputs)
        {
            List<ResultBoxWithMasks> boxes = base.DecodeNNOut(outputs);
            return boxes; 
        }


        protected override ResultBoxWithMasks DecodeBox(float[,] array, int box)
        {
            ResultBox boxResult = ResultBox.DecodeBox(array, box, new());

            float[] masksScore = Array2DTo1DCopy(array, box, 4 + ClassesNum, 32);
            Tensor masksScoresTensor = new(1, masksScore.Length, masksScore);

            ResultBoxWithMasks result = new(boxResult, null, masksScoresTensor);
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