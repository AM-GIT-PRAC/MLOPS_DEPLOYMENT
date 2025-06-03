
import mlflow
import os
import sys

def cleanup_old_runs():
    """Clean up old MLflow runs based on environment configuration"""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    try:
        # Get configuration from environment
        experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'fraud-detection-production')
        max_runs = int(os.getenv('MAX_MLFLOW_RUNS_TO_KEEP', '5'))
        
        print(f"🔍 Looking for experiment: {experiment_name}")
        print(f"📊 Max runs to keep: {max_runs}")
        
        # Get or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
        except Exception:
            experiment = None
        
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            print(f"📈 Found {len(runs)} existing runs")
            
            if len(runs) > max_runs:
                old_runs = runs.iloc[max_runs:]
                deleted_count = 0
                
                for _, run in old_runs.iterrows():
                    try:
                        mlflow.delete_run(run.run_id)
                        print(f"🗑️ Deleted old run: {run.run_id}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"⚠️ Could not delete run {run.run_id}: {e}")
                
                print(f"✅ Deleted {deleted_count} old runs")
            
            print(f"✅ MLflow cleanup completed. Keeping {min(len(runs), max_runs)} runs.")
        else:
            print("📝 Experiment not found, will be created during training.")
            
    except Exception as e:
        print(f"⚠️ MLflow cleanup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    cleanup_old_runs()
