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

        IOps ops;
        
        public YOLOv8SegmentationPostprocessor()
        {
            ops = BarracudaUtils.CreateOps(WorkerFactory.Type.ComputePrecompiled);
        }


        protected override List<ResultBoxWithMasks> DecodeNNOut(Tensor[] outputs)
        {
            List<ResultBoxWithMasks> boxes = base.DecodeNNOut(outputs);
            boxes = DecodeMasks(outputs[1], boxes);
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

        private List<ResultBoxWithMasks> DecodeMasks(Tensor masks, List<ResultBoxWithMasks> results)
        {
            if (results.Count == 0)
                return results;

            var allMaskScoresArray = results.Select(box => box.maskInd).ToArray();
            Tensor allMaskScoresTensor = ops.Concat(allMaskScoresArray, axis: 0);
            Tensor allMaskScoresReshaped = ops.Reshape(allMaskScoresTensor, new TensorShape(results.Count, 1, 1, allMaskScoresTensor.channels));
            allMaskScoresTensor.tensorOnDevice.Dispose();
            Tensor boxMasks = ops.Mul(new[] { masks, allMaskScoresReshaped });
            allMaskScoresReshaped.tensorOnDevice.Dispose();
            Tensor reducedBoxMask = ops.ReduceSum(boxMasks, axis: -1);
            boxMasks.tensorOnDevice.Dispose();
            Tensor boxMasks1 = ops.Sigmoid(reducedBoxMask);
            reducedBoxMask.tensorOnDevice.Dispose();
            boxMasks = ops.Upsample2D(boxMasks1, new[] { 4, 4 }, true);
            boxMasks1.tensorOnDevice.Dispose();

            for (int i = 0; i < results.Count; i++)
            {
                ResultBoxWithMasks box = results[i];
                Tensor maskSlice = ops.StridedSlice(boxMasks, new[] { i, (int)box.rect.yMin, (int)box.rect.xMin, 0 }, new[] { i + 1, (int)box.rect.yMax, (int)box.rect.xMax, boxMasks.channels }, new[] { 1, 1, 1, 1 });
                int xEndPad = boxMasks.width - (int)box.rect.xMin - maskSlice.width;
                int yEndPad = boxMasks.height - (int)box.rect.yMin - maskSlice.height;
                Tensor padded = ops.Border2D(maskSlice, new[] { (int)box.rect.xMin, (int)box.rect.yMin, 0, xEndPad, yEndPad, 0 }, 0);

                box.masks = padded;
                maskSlice.tensorOnDevice.Dispose();
                results[i].maskInd.tensorOnDevice.Dispose();
            }
            boxMasks.tensorOnDevice.Dispose();
            return results;
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