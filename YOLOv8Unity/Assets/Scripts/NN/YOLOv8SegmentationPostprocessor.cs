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
        private IOps ops;
        
        public YOLOv8SegmentationPostprocessor()
        {
            outputReader = new YOLOv8SegmentationOutputReader();
            ops = BarracudaUtils.CreateOps(WorkerFactory.Type.ComputePrecompiled);
        }


        public new List<ResultBoxWithMasks> Postprocess(Tensor[] outputs)
        {
            Profiler.BeginSample("YOLOv8SegmentationPostprocessor.Postprocess");
            List<ResultBox> boxes = base.Postprocess(outputs);

            Tensor masksOutput = outputs[1];
            List<ResultBoxWithMasksIndices> boxesWithIndices = boxes.Select(box => (ResultBoxWithMasksIndices)box).ToList();
            List<ResultBoxWithMasks> boxesWithMasks = DecodeMasks(masksOutput, boxesWithIndices);
            Profiler.EndSample();
            return boxesWithMasks;
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
                results.Add(r);

                maskSlice.tensorOnDevice.Dispose();
                box.maskInd.tensorOnDevice.Dispose();
            }
            boxMasks.tensorOnDevice.Dispose();
            return results;
        }


    }
}