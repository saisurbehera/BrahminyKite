"""
Performance benchmarks for BrahminyKite tools.
"""

import time
import asyncio
import statistics
from typing import List, Dict, Any

from ..clients import EmpiricalToolsClient
from ..service_manager import ServiceManager, ServiceClient


class ToolsBenchmark:
    """Benchmark tool performance."""
    
    def __init__(self):
        self.manager = ServiceManager()
        self.client = ServiceClient()
        self.results = {}
    
    def setup(self):
        """Start services for benchmarking."""
        print("Starting services...")
        self.manager.start_all()
        time.sleep(2)  # Wait for services to start
    
    def teardown(self):
        """Stop services after benchmarking."""
        self.manager.stop_all()
        self.client.close_all()
    
    def benchmark_empirical_z3(self, iterations: int = 1000):
        """Benchmark Z3 logical consistency checking."""
        client = EmpiricalToolsClient()
        
        formulas = [
            "(and (> x 0) (< x 10))",
            "(or (= x 5) (= y 10))",
            "(and (> (* x x) 16) (< x 5))",
        ]
        
        times = []
        
        for i in range(iterations):
            formula = formulas[i % len(formulas)]
            start = time.perf_counter()
            
            result = client.check_logical_consistency(formula)
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        client.close()
        
        return {
            'mean_ms': statistics.mean(times),
            'median_ms': statistics.median(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'p95_ms': statistics.quantiles(times, n=20)[18],  # 95th percentile
            'throughput': iterations / sum(times) * 1000
        }
    
    def benchmark_empirical_duckdb(self, iterations: int = 1000):
        """Benchmark DuckDB queries."""
        client = EmpiricalToolsClient()
        
        queries = [
            "SELECT COUNT(*) FROM facts WHERE confidence > 0.8",
            "SELECT * FROM facts ORDER BY timestamp DESC LIMIT 10",
            "SELECT source, AVG(confidence) FROM facts GROUP BY source",
        ]
        
        times = []
        
        for i in range(iterations):
            query = queries[i % len(queries)]
            start = time.perf_counter()
            
            result = client.query_facts(query)
            
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        client.close()
        
        return {
            'mean_ms': statistics.mean(times),
            'median_ms': statistics.median(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'p95_ms': statistics.quantiles(times, n=20)[18],
            'throughput': iterations / sum(times) * 1000
        }
    
    async def benchmark_batch_processing(self, batch_size: int = 100):
        """Benchmark batch processing capabilities."""
        client = EmpiricalToolsClient()
        
        formulas = [f"(> x {i})" for i in range(batch_size)]
        
        start = time.perf_counter()
        results = await client.batch_check_consistency(formulas)
        end = time.perf_counter()
        
        client.close()
        
        total_time_ms = (end - start) * 1000
        
        return {
            'total_time_ms': total_time_ms,
            'per_item_ms': total_time_ms / batch_size,
            'throughput': batch_size / total_time_ms * 1000
        }
    
    async def benchmark_parallel_frameworks(self):
        """Benchmark parallel framework execution."""
        claim = "The Earth is approximately 4.5 billion years old"
        
        start = time.perf_counter()
        results = await self.client.verify_claim(claim)
        end = time.perf_counter()
        
        return {
            'frameworks': list(results.keys()),
            'total_time_ms': (end - start) * 1000,
            'results': results
        }
    
    def run_all_benchmarks(self):
        """Run all benchmarks."""
        print("\n=== BrahminyKite Tools Performance Benchmark ===\n")
        
        self.setup()
        
        try:
            # Empirical Z3 benchmark
            print("1. Z3 Logical Consistency (1000 iterations):")
            z3_results = self.benchmark_empirical_z3()
            for key, value in z3_results.items():
                print(f"   {key}: {value:.2f}")
            
            # DuckDB benchmark
            print("\n2. DuckDB Queries (1000 iterations):")
            db_results = self.benchmark_empirical_duckdb()
            for key, value in db_results.items():
                print(f"   {key}: {value:.2f}")
            
            # Batch processing benchmark
            print("\n3. Batch Processing (100 items):")
            batch_results = asyncio.run(self.benchmark_batch_processing())
            for key, value in batch_results.items():
                print(f"   {key}: {value:.2f}")
            
            # Parallel frameworks benchmark
            print("\n4. Parallel Framework Execution:")
            parallel_results = asyncio.run(self.benchmark_parallel_frameworks())
            print(f"   Frameworks: {parallel_results['frameworks']}")
            print(f"   Total time: {parallel_results['total_time_ms']:.2f} ms")
            
            # Summary
            print("\n=== Summary ===")
            print(f"✓ Z3 throughput: {z3_results['throughput']:.0f} ops/sec")
            print(f"✓ DuckDB throughput: {db_results['throughput']:.0f} queries/sec")
            print(f"✓ Batch throughput: {batch_results['throughput']:.0f} items/sec")
            print(f"✓ All benchmarks completed successfully")
            
        finally:
            self.teardown()


def compare_with_baseline():
    """Compare with baseline performance."""
    print("\n=== Performance Comparison ===\n")
    
    # Baseline: Direct library calls without gRPC
    import z3
    
    # Direct Z3 benchmark
    times = []
    for i in range(100):
        start = time.perf_counter()
        
        solver = z3.Solver()
        x = z3.Int('x')
        solver.add(x > 0, x < 10)
        solver.check()
        
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    direct_mean = statistics.mean(times)
    
    # gRPC benchmark
    benchmark = ToolsBenchmark()
    benchmark.setup()
    grpc_results = benchmark.benchmark_empirical_z3(iterations=100)
    benchmark.teardown()
    
    print(f"Direct Z3 mean: {direct_mean:.2f} ms")
    print(f"gRPC Z3 mean: {grpc_results['mean_ms']:.2f} ms")
    print(f"Overhead: {(grpc_results['mean_ms'] / direct_mean - 1) * 100:.1f}%")


if __name__ == '__main__':
    benchmark = ToolsBenchmark()
    benchmark.run_all_benchmarks()
    
    # Run comparison
    compare_with_baseline()