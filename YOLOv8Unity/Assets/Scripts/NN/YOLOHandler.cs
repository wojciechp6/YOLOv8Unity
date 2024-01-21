using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Profiling;

namespace NN
{
    public class YOLOHandler
    {
        NNHandler nn;
        public ReferenceComputeOps ops;

        Tensor premulTensor;

        public YOLOHandler(NNHandler nn)
        {
            this.nn = nn;
            ops = BarracudaUtils.CreateOps(WorkerFactory.Type.ComputePrecompiled) as ReferenceComputeOps;
        }

        public List<ResultBoxWithMasks> Run(Texture2D tex)
        {
            Profiler.BeginSample("YOLO.Run");

            Tensor input = new Tensor(tex);
            var preprocessed = Preprocess(input);
            Execute(preprocessed);
            List<Tensor> outputs = PeekOutputs();
            var results = Postprocess(outputs[0]);

            Tensor masks = outputs[1];
            input.Dispose();

            if (results.Count == 0)
                return results;

            var allMaskScoresArray = results.Select(box => box.maskInd).ToArray();
            Tensor allMaskScoresTensor = ops.Concat(allMaskScoresArray, axis: 0);
            Tensor allMaskScoresReshaped = ops.Reshape(allMaskScoresTensor, new TensorShape(results.Count, 1, 1, allMaskScoresTensor.channels));
            allMaskScoresTensor.tensorOnDevice.Dispose();
            Tensor boxMasks = ops.Mul(new[] {masks, allMaskScoresReshaped});
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

            masks.Dispose();

            Profiler.EndSample();
            return results;
        }

        private void Execute(Tensor preprocessed)
        {
            Profiler.BeginSample("YOLO.Execute");

            nn.worker.Execute(preprocessed);
            nn.worker.FlushSchedule(blocking: true);
            Profiler.EndSample();

        }

        private List<Tensor> PeekOutputs()
        {
            List<Tensor> outputs = new();

            foreach (string outputName in nn.model.outputs)
            {
                Tensor output = nn.worker.PeekOutput(outputName);
                outputs.Add(output);

            }
            return outputs;
        }

        private Tensor Preprocess(Tensor x)
        {
            Profiler.BeginSample("YOLO.Preprocess");
            var preprocessed = x;
            //var preprocessed = ops.Mul(new[]{ x, premulTensor });
            //var preprocessed = ops.Transpose(x, new[] {0, 3, 1, 2 });
            Profiler.EndSample();
            return preprocessed;
        }

        List<ResultBoxWithMasks> Postprocess(Tensor x)
        {
            Profiler.BeginSample("YOLO.Postprocess");
            var results = YOLOv8Postprocessor.DecodeNNOut(x);
            results = DuplicatesSupressor.RemoveDuplicats(results);
            Profiler.EndSample();
            return results;
        }
    }
}