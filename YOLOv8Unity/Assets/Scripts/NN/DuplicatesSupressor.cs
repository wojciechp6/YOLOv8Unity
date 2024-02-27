using NN;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine.Profiling;

public static class DuplicatesSupressor
{
    const float OVERLAP_TRESHOLD = 0.3f;

    static public List<T> RemoveDuplicats<T>(List<T> boxes) where T : ResultBox
    {
        Profiler.BeginSample("DuplicatesSupressor.RemoveDuplicats");

        if (boxes.Count == 0)
            return boxes;

        List<T> result_boxes = new();

        for (int classIndex = 0; classIndex < 80; classIndex++)
        {

            var classBoxes = boxes.Where(box => box.bestClassIndex == classIndex).ToList();
            RemoveDuplicatesForClass(classBoxes);
            classBoxes = classBoxes.Where(box => box.score > 0).ToList();
            result_boxes.AddRange(classBoxes);
        }

        Profiler.EndSample();

        return result_boxes;
    }

    private static void RemoveDuplicatesForClass<T>(List<T> boxes) where T : ResultBox
    {
        SortBoxesByScore(boxes);
        for (int i = 0; i < boxes.Count; i++)
        {
            T i_box = boxes[i];
            if (i_box.score == 0)
                continue;

            for (int j = i + 1; j < boxes.Count; j++)
            {
                T j_box = boxes[j];
                float iou = IntersectionOverUnion.CalculateIOU(i_box.rect, j_box.rect);
                if (iou >= OVERLAP_TRESHOLD && i_box.score > j_box.score)
                {
                    j_box.score = 0;
                    boxes[j] = j_box;
                }
            }
        }
    }

    private static List<T> SortBoxesByScore<T>(List<T> boxes) where T : ResultBox
    {
        Comparison<ResultBox> boxClassValueComparer =
            (box1, box2) => box2.score.CompareTo(box1.score);
        boxes.Sort(boxClassValueComparer);
        return boxes;
    }
}