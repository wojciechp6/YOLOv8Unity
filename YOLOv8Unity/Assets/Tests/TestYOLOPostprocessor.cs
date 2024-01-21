//using NN;
//using NUnit.Framework;
//using System.Collections.Generic;
//using Unity.Barracuda;

//namespace Tests
//{
//    public class TestYOLOPostprocessor
//    {
//        public Tensor CreateInputTensorWithGivenParameters(float[] scores, int[] classIndexes)
//        {
//            const int boxScoreIndex = 4;
//            const int classesOffset = 5;

//            const int boxesPerCell = 5;
//            const int boxSize = 25;

//            Assert.AreEqual(scores.Length, classIndexes.Length, "Scores and classIndexes should have same lenght.");
//            Assert.LessOrEqual(scores.Length, boxesPerCell, "Scores and classIndexes lenght can't be larger than boxesPerCell number");

//            float[,,,] testInput = new float[1, 1, 1, 125];

//            for (int box = 0; box < scores.Length; box++)
//            {
//                int targetClassIndex = classIndexes[box];
//                int offset = box * boxSize;
//                testInput[0, 0, 0, offset + boxScoreIndex] = scores[box];
//                testInput[0, 0, 0, offset + classesOffset + targetClassIndex] = 0.9f;
//            }

//            Tensor testTensor = new(new[] { 1, 1, 1, 125 }, testInput);
//            return testTensor;
//        }

//        public Tensor CreateInputTensorWithDefaultParameters()
//        {
//            return CreateInputTensorWithGivenParameters(new[] { 100f }, new[] { 8 });
//        }

//        [Test]
//        public void ReadsTensorSuccessfully()
//        {
//            Tensor testTensor = CreateInputTensorWithDefaultParameters();
//            List<ResultBox> results = YOLOv2Postprocessor.DecodeNNOut(testTensor);

//            Assert.AreEqual(1, results.Count);
//        }

//        [Test]
//        public void RightBestClass()
//        {
//            const int targetClassIndex = 10;
//            Tensor testTensor = CreateInputTensorWithGivenParameters(new[] { 100f }, new[] { targetClassIndex });

//            List<ResultBox> results = YOLOv2Postprocessor.DecodeNNOut(testTensor);
//            Assert.AreEqual(targetClassIndex, results[0].bestClassIndex);
//        }

//        [Test]
//        public void RemovesBoxWithLowScore()
//        {
//            const int targetClassIndex = 10;
//            const float targetScore = -1000f;
//            Tensor testTensor = CreateInputTensorWithGivenParameters(new[] { targetScore }, new[] { targetClassIndex });

//            List<ResultBox> results = YOLOv2Postprocessor.DecodeNNOut(testTensor);
//            Assert.AreEqual(0, results.Count);
//        }
//    }
//}