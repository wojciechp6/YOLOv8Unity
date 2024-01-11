using UnityEngine;

public static class IntersectionOverUnion
{
    public static float CalculateIOU(Rect box1, Rect box2)
    {
        float intersect_w = IntervalOverlap(box1.xMin, box1.xMax, box2.xMin, box2.xMax);
        float intersect_h = IntervalOverlap(box1.yMin, box1.yMax, box2.yMin, box2.yMax);

        float intersect = intersect_w * intersect_h;

        float union = box1.width * box1.height + box2.width * box2.height - intersect;
        return intersect / union;
    }

    static float IntervalOverlap(float box1_min, float box1_max, float box2_min, float box2_max)
    {
        if (box2_min < box1_min)
        {
            if (box2_max < box1_min)
                return 0;
            else
                return Mathf.Min(box1_max, box2_max) - box1_min;
        }
        else
        {
            if (box1_max < box2_min)
                return 0;
            else
                return Mathf.Min(box1_max, box2_max) - box2_min;
        }
    }
}
