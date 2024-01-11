using Unity.Barracuda;
using UnityEngine;

namespace NN
{
    public class ResultBox
    {
        public Rect rect;
        public float confidence;
        public float[] classes;
        public int bestClassIdx;
        public Tensor maskInd;
        public Tensor masks;

        ~ResultBox()
        {
            maskInd.tensorOnDevice.Dispose();
            masks.tensorOnDevice.Dispose();
        }
    }
}