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

        public YOLOHandler(NNHandler nn)
        {
            this.nn = nn;
        }

        public List<ResultBoxWithMasks> Run(Texture2D tex)
        {
            Profiler.BeginSample("YOLO.Run");
            Tensor input = new Tensor(tex);
            var preprocessed = Preprocess(input);
            Execute(preprocessed);
            List<Tensor> outputs = PeekOutputs();
            var results = Postprocess(outputs);
            input.tensorOnDevice.Dispose();
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
            Profiler.EndSample();
            return preprocessed;
        }

        List<ResultBoxWithMasks> Postprocess(List<Tensor> x)
        {
            Profiler.BeginSample("YOLO.Postprocess");
            var results = new YOLOv8SegmentationPostprocessor().Postprocess(x.ToArray());
            Profiler.EndSample();
            return results;
        }
    }
}