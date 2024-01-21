using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Profiling;

namespace NN
{
    public class YOLOv8SegmentationPostprocessor : YOLOv8Postprocessor
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


        public new List<ResultBoxWithMasks> Postprocess(Tensor[] outputs)
        {
            Profiler.BeginSample("YOLOv8Postprocessor.Postprocess");
            List<ResultBox> boxes = DecodeNNOut(outputs);
            Profiler.EndSample();
            return boxes.Select(box => box as ResultBoxWithMasks).ToList();
        }

        protected override List<ResultBox> DecodeNNOut(Tensor[] outputs)
        {
            List<ResultBoxWithMasksIndices> boxes = base.DecodeNNOut(outputs).Select(box => box as ResultBoxWithMasksIndices).ToList();
            List<ResultBoxWithMasks> boxesMasks = DecodeMasks(outputs[1], boxes);
            List<ResultBox> boxes1 = boxesMasks.Select(box => box as ResultBox).ToList();
            return boxes1; 
        }


        protected override ResultBox DecodeBox(float[,] array, int box)
        {
            ResultBox boxResult = ResultBox.DecodeBox(array, box, new());

            float[] masksScore = Array2DTo1DCopy(array, box, 4 + ClassesNum, 32);
            Tensor masksScoresTensor = new(1, masksScore.Length, masksScore);

            ResultBoxWithMasks result = new(boxResult, masksScoresTensor);
            return result;
        }

        private List<ResultBoxWithMasks> DecodeMasks(Tensor masks, List<ResultBoxWithMasksIndices> boxes)
        {
            List<ResultBoxWithMasks> results = new();
            if (boxes.Count == 0)
                return results;

            var allMaskScoresArray = boxes.Select(box => box.maskInd).ToArray();
            Tensor allMaskScoresTensor = ops.Concat(allMaskScoresArray, axis: 0);
            Tensor allMaskScoresReshaped = ops.Reshape(allMaskScoresTensor, new TensorShape(boxes.Count, 1, 1, allMaskScoresTensor.channels));
            allMaskScoresTensor.tensorOnDevice.Dispose();
            Tensor boxMasks = ops.Mul(new[] { masks, allMaskScoresReshaped });
            allMaskScoresReshaped.tensorOnDevice.Dispose();
            Tensor reducedBoxMask = ops.ReduceSum(boxMasks, axis: -1);
            boxMasks.tensorOnDevice.Dispose();
            Tensor boxMasks1 = ops.Sigmoid(reducedBoxMask);
            reducedBoxMask.tensorOnDevice.Dispose();
            boxMasks = ops.Upsample2D(boxMasks1, new[] { 4, 4 }, true);
            boxMasks1.tensorOnDevice.Dispose();

            for (int i = 0; i < boxes.Count; i++)
            {
                ResultBoxWithMasksIndices box = boxes[i];
                Tensor maskSlice = ops.StridedSlice(boxMasks, new[] { i, (int)box.rect.yMin, (int)box.rect.xMin, 0 }, new[] { i + 1, (int)box.rect.yMax, (int)box.rect.xMax, boxMasks.channels }, new[] { 1, 1, 1, 1 });
                int xEndPad = boxMasks.width - (int)box.rect.xMin - maskSlice.width;
                int yEndPad = boxMasks.height - (int)box.rect.yMin - maskSlice.height;
                Tensor padded = ops.Border2D(maskSlice, new[] { (int)box.rect.xMin, (int)box.rect.yMin, 0, xEndPad, yEndPad, 0 }, 0);

                ResultBoxWithMasks r = new(box, padded);


                maskSlice.tensorOnDevice.Dispose();
                boxes[i].maskInd.tensorOnDevice.Dispose();
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