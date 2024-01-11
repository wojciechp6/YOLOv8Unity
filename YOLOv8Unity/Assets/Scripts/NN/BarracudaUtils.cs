using Unity.Barracuda;

public class BarracudaUtils
{

    public static IOps CreateOps(WorkerFactory.Type type, bool verbose = false)
    {
        WorkerFactory.ValidateType(type);
        switch (type)
        {
            case WorkerFactory.Type.ComputePrecompiled:
                return new PrecompiledComputeOps(verbose: verbose);

            case WorkerFactory.Type.Compute:
                return new ComputeOps(verbose: verbose);

            case WorkerFactory.Type.ComputeRef:
                return new ReferenceComputeOps();

            case WorkerFactory.Type.CSharp:
                return new UnsafeArrayCPUOps();

            default:
                return new ReferenceCPUOps();
        }
    }
}
