using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

static class Extensions
{
    public static string ArrayToString(this IEnumerable enumerable)
    {
        string str = "[";
        foreach (var e in enumerable)
        {
            str += e;
        }
        str += "]";
        return str;
    }

    public static T[] GetRange<T>(this ICollection<T> collection, int start = 0, int end = -1)
    {
        end = end < 0 ? end = collection.Count + end + 1 : end;
        var arr = Array.CreateInstance(typeof(T), end - start);
        for (int i = start, j = 0; i < end; i++, j++)
        {
            var v = collection.ElementAt(i);
            arr.SetValue(v, j);
        }
        return (T[])arr;
    }

    public static int MaxIdx(this ICollection<float> collection)
    {
        int idx = 0;
        float max = collection.ElementAt(0);
        for (int i = 1; i < collection.Count; i++)
        {
            if (collection.ElementAt(i) > max)
            {
                idx = i;
                max = collection.ElementAt(i);
            }
        }
        return idx;
    }

    public static void ForEach<T>(this IEnumerable<T> enumerable, Action<T> func)
    {
        foreach (var e in enumerable)
            func(e);
    }

    public static void ForEach<T>(this ICollection<T> collection, Action<T, int> func)
    {
        for (int i = 0; i < collection.Count; i++)
            func(collection.ElementAt(i), i);
    }

    public static IList<T> Update<T>(this IList<T> collection, Func<T, T> func)
    {
        for (int i = 0; i < collection.Count; i++)
            collection[i] = func(collection[i]);
        return collection;
    }

    public static IList<T> Update<T>(this IList<T> collection, Func<T, int, T> func)
    {
        for (int i = 0; i < collection.Count; i++)
            collection[i] = func(collection[i], i);
        return collection;
    }
}
