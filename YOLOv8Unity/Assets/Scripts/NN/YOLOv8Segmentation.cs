using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Profiling;

namespace NN
{
    public class YOLOv8Segmentation : YOLOv8
    {
        private IOps ops;
        
        public YOLOv8Segmentation(NNHandler nn) : base(nn)
        {
            outputReader = new YOLOv8SegmentationOutputReader();
            ops = BarracudaUtils.CreateOps(WorkerFactory.Type.ComputePrecompiled);
        }

        public new List<ResultBoxWithMask> Run(Texture2D image)
        {
            Profiler.BeginSample("YOLO.Run");
            var outputs = ExecuteModel(image);
            var results = Postprocess(outputs);
            Profiler.EndSample();
            return results;
        }

        protected new List<ResultBoxWithMask> Postprocess(Tensor[] outputs)
        {
            Profiler.BeginSample("YOLOv8SegmentationPostprocessor.Postprocess");
            List<ResultBox> boxes = base.Postprocess(outputs);

            Tensor masksOutput = outputs[1];
            List<ResultBoxWithMasksIndices> boxesWithIndices = boxes.Select(box => (ResultBoxWithMasksIndices)box).ToList();
            List<ResultBoxWithMask> boxesWithMasks = DecodeMasks(masksOutput, boxesWithIndices);
            Profiler.EndSample();
            return boxesWithMasks;
        }

        private List<ResultBoxWithMask> DecodeMasks(Tensor masks, List<ResultBoxWithMasksIndices> boxes)
        {
            Profiler.BeginSample("YOLOv8SegmentationPostprocessor.DecodeMasks");

            if (boxes.Count == 0)
                return new();

            var allMaskScoresArray = boxes.Select(box => box.maskInd).ToArray();
            Tensor allMaskScoresTensor = ops.Concat(allMaskScoresArray, axis: 0);
            boxes.ForEach(box => box.maskInd.tensorOnDevice.Dispose());

            Tensor allMaskScoresReshaped = ops.Reshape(allMaskScoresTensor, new TensorShape(boxes.Count, 1, 1, allMaskScoresTensor.channels));
            allMaskScoresTensor.tensorOnDevice.Dispose();

            Tensor boxMasks = ops.Mul(new[] { masks, allMaskScoresReshaped });
            allMaskScoresReshaped.tensorOnDevice.Dispose();

            Tensor reducedBoxMask = ops.ReduceSum(boxMasks, axis: -1);
            boxMasks.tensorOnDevice.Dispose();

            Tensor downscaledMasks = ops.Sigmoid(reducedBoxMask);
            reducedBoxMask.tensorOnDevice.Dispose();

            int[] downscaleFactor = new[] { 4, 4 };
            boxMasks = ops.Upsample2D(downscaledMasks, downscaleFactor, true);
            downscaledMasks.tensorOnDevice.Dispose();

            List<ResultBoxWithMask> resultMasks = SeparateAndCutMasks(boxes, boxMasks).ToList();
            boxMasks.tensorOnDevice.Dispose();

            Profiler.EndSample();
            return resultMasks;
        }

        private IEnumerable<ResultBoxWithMask> SeparateAndCutMasks(List<ResultBoxWithMasksIndices> boxes, Tensor boxMasks)
        {
            for (int i = 0; i < boxes.Count; i++)
            {
                ResultBoxWithMasksIndices box = boxes[i];
                RectInt rect = new RectInt((int)box.rect.xMin, (int)box.rect.yMin, (int)box.rect.width, (int)box.rect.height);

                int[] startIndexes = new[] { i, rect.yMin, rect.xMin, 0 };
                int[] stopIndexes = new[] { i + 1, rect.yMax, rect.xMax, boxMasks.channels };
                int[] strides = new[] { 1, 1, 1, 1 };
                Tensor maskSlice = ops.StridedSlice(boxMasks, startIndexes, stopIndexes, strides);

                int xEndPad = boxMasks.width - rect.xMin - maskSlice.width;
                int yEndPad = boxMasks.height - rect.yMin - maskSlice.height;
                int[] padsSize = new[] { rect.xMin, rect.yMin, 0, xEndPad, yEndPad, 0 };
                Tensor padded = ops.Border2D(maskSlice, padsSize, 0);

                ResultBoxWithMask resultMask = new(box, padded);

                maskSlice.tensorOnDevice.Dispose();
                box.maskInd.tensorOnDevice.Dispose();

                yield return resultMask;
            }
        }
    }
}