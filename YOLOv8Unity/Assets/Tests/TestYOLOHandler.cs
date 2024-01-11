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
        const float min_confidence = 0.1f;
        private YOLOHandler yolo;
        private Texture2D test_image;

        [SetUp]
        public void Setup()
        {
            NNModel model = AssetDatabase.LoadAssetAtPath<NNModel>(MODEL_PATH);
            NNHandler nnHandler = new(model);
            yolo = new YOLOHandler(nnHandler);
            test_image = AssetDatabase.LoadAssetAtPath<Texture2D>(IMAGE_PATH);
        }

        [TearDown]
        public void TearDown()
        {
            yolo.Dispose();
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
            int firstExpectedClass = 19;
            int secondExpectedClass = 14;

            // when
            var results = yolo.Run(test_image);
            var confident_results = GetConfidentResults(results);

            // then
            Assert.AreEqual(firstExpectedClass, confident_results[0].bestClassIdx);
            Assert.AreEqual(secondExpectedClass, confident_results[1].bestClassIdx);
        }

        [Test]
        public void ConfidentResultsHasRightBoxes()
        {
            // given
            Rect firstExpectedRect = new(x: 0.56f, y: 0.20f, width: 0.23f, height: 0.31f);
            Rect secondExpectedRect = new(x: -0.01f, y: 0.13f, width: 0.55f, height: 0.76f);

            // when
            var results = yolo.Run(test_image);
            var confident_results = GetConfidentResults(results);

            // then
            AssertAreRectsEqual(firstExpectedRect, confident_results[0].rect);
            AssertAreRectsEqual(secondExpectedRect, confident_results[1].rect);
        }

        private List<ResultBox> GetConfidentResults(List<ResultBox> rawResults)
        {
            return rawResults.Where(box => box.classes[box.bestClassIdx] > min_confidence).ToList();
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