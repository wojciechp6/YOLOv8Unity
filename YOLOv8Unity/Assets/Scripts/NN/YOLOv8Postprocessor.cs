using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Profiling;

namespace NN
{
    public abstract class YOLOv8Postprocessor<T> where T : ResultBox
    {
        public float DiscardThreshold = 0.1f;
        const int ClassesNum = 80;
        const int BoxesPerCell = 8400;
        const int WidthHeight = 640;

        public List<T> Postprocess(Tensor[] outputs)
        {
            Profiler.BeginSample("YOLOv8Postprocessor.Postprocess");
            List<T> boxes = DecodeNNOut(outputs);
            Profiler.EndSample();
            return boxes;
        }

        protected virtual List<T> DecodeNNOut(Tensor[] outputs)
        {
            Tensor firstOutput = outputs[0];
            float[,] array = ReadOutputToArray(firstOutput);
            List<T> boxes = DecodeBoxes(array).ToList();
            boxes = DuplicatesSupressor.RemoveDuplicats(boxes);
            return boxes;
        }

        private float[,] ReadOutputToArray(Tensor output)
        {
            var reshapedOutput = output.Reshape(new[] { 1, 1, BoxesPerCell, -1 });
            var array = TensorToArray2D(reshapedOutput);
            reshapedOutput.Dispose();
            return array;
        }

        private IEnumerable<T> DecodeBoxes(float[,] array)
        {
            int boxes = array.GetLength(0);
            for (int box_index = 0; box_index < boxes; box_index++)
            {
                T box = DecodeBox(array, box_index);
                if (box != null)
                    yield return box;
            }
        }

        protected abstract T DecodeBox(float[,] array, int box);

        private float[,] TensorToArray2D(Tensor tensor)
        {
            float[,] output = new float[tensor.width, tensor.channels];
            var data = tensor.AsFloats();
            int bytes = Buffer.ByteLength(data);
            Buffer.BlockCopy(data, 0, output, 0, bytes);
            return output;
        }
    }
}