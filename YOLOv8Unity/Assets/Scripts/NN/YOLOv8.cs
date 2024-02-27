using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Profiling;

namespace NN
{
    public class YOLOv8
    {
        protected YOLOv8OutputReader outputReader;

        private NNHandler nn;

        public YOLOv8(NNHandler nn)
        {
            this.nn = nn;
            outputReader = new();
        }

        public List<ResultBox> Run(Texture2D image)
        {
            Profiler.BeginSample("YOLO.Run");
            var outputs = ExecuteModel(image);
            var results = Postprocess(outputs);
            Profiler.EndSample();
            return results;
        }

        protected Tensor[] ExecuteModel(Texture2D image)
        {
            Tensor input = new Tensor(image);
            ExecuteBlocking(input);
            input.tensorOnDevice.Dispose();
            return PeekOutputs().ToArray();
        }

        private void ExecuteBlocking(Tensor preprocessed)
        {
            Profiler.BeginSample("YOLO.Execute");
            nn.worker.Execute(preprocessed);
            nn.worker.FlushSchedule(blocking: true);
            Profiler.EndSample();
        }

        private IEnumerable<Tensor> PeekOutputs()
        {
            foreach (string outputName in nn.model.outputs)
            {
                Tensor output = nn.worker.PeekOutput(outputName);
                yield return output;
            }
        }

        protected List<ResultBox> Postprocess(Tensor[] outputs)
        {
            Profiler.BeginSample("YOLOv8Postprocessor.Postprocess");
            Tensor boxesOutput = outputs[0];
            List<ResultBox> boxes = outputReader.ReadOutput(boxesOutput).ToList();
            boxes = DuplicatesSupressor.RemoveDuplicats(boxes);
            Profiler.EndSample();
            return boxes;
        }
    }
}