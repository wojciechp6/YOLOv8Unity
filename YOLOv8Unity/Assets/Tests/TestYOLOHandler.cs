using NN;
using NUnit.Framework;
using System.Collections.Generic;
using System.Linq;
using Unity.Barracuda;
using UnityEditor;
using UnityEngine;

namespace Tests
{
    public class TestYOLOHandler
    {
        const string MODEL_PATH = "Assets/YOLOv2 Tiny.onnx";
        const string IMAGE_PATH = "Assets/Tests/test_image.jpg";
        const float min_confidence = 0.15f;
        private YOLOv8 yolo;
        NNHandler nnHandler;
        private Texture2D test_image;

        [SetUp]
        public void Setup()
        {
            NNModel model = AssetDatabase.LoadAssetAtPath<NNModel>(MODEL_PATH);
            nnHandler = new(model);
            yolo = new YOLOv8(nnHandler);
            test_image = AssetDatabase.LoadAssetAtPath<Texture2D>(IMAGE_PATH);
        }

        [TearDown]
        public void TearDown()
        {
            nnHandler.Dispose();
        }

        [Test]
        public void CreatesSuccessfully()
        {
            Assert.NotNull(yolo);
        }

        [Test]
        public void NotZeroResult()
        {
            // when
            var results = yolo.Run(test_image);
            Assert.NotZero(results.Count);
        }

        [Test]
        public void TwoConfidentResults()
        {
            // when
            var results = yolo.Run(test_image);
            var confident_results = GetConfidentResults(results);

            // then
            Assert.AreEqual(2, confident_results.Count);
        }

        [Test]
        public void ConfidentResultsHasRightClasses()
        {
            // given 
            int firstExpectedClass = 14;
            int secondExpectedClass = 19;

            // when
            var results = yolo.Run(test_image);
            var confident_results = GetConfidentResults(results);

            // then
            Assert.AreEqual(firstExpectedClass, confident_results[0].bestClassIndex);
            Assert.AreEqual(secondExpectedClass, confident_results[1].bestClassIndex);
        }

        [Test]
        public void ConfidentResultsHasRightBoxes()
        {
            // given
            Rect firstExpectedRect = new(x: -3.34f, y: 53.32f, width: 229.38f, height: 318.09f);
            Rect secondExpectedRect = new(x:234.16f, y:83.54f, width:94.07f, height:129.21f);

            // when
            var results = yolo.Run(test_image);
            var confident_results = GetConfidentResults(results);

            // then
            AssertAreRectsEqual(firstExpectedRect, confident_results[0].rect);
            AssertAreRectsEqual(secondExpectedRect, confident_results[1].rect);
        }

        private List<T> GetConfidentResults<T>(List<T> rawResults) where T : ResultBox
        {
            return rawResults.Where(box => box.score > min_confidence).ToList();
        }

        private void AssertAreRectsEqual(Rect expected, Rect actual)
        {
            const float delta = 0.01f;
            Assert.AreEqual(expected.xMin, actual.xMin, delta);
            Assert.AreEqual(expected.xMax, actual.xMax, delta);
            Assert.AreEqual(expected.yMin, actual.yMin, delta);
            Assert.AreEqual(expected.yMax, actual.yMax, delta);
        }

    }
}