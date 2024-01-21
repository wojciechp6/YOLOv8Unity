using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Profiling;

namespace NN
{
    public class YOLOv8Postprocessor
    {
        protected YOLOv8OutputReader outputReader;

        public YOLOv8Postprocessor()
        {
            outputReader = new();
        }

        public List<ResultBox> Postprocess(Tensor[] outputs)
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