import mlflow
import os
import sys
from datetime import datetime, timedelta

def cleanup_old_runs():
    """Clean up old MLflow runs based on environment configuration"""
    
    # Get MLflow tracking URI
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
    mlflow.set_tracking_uri(mlflow_uri)
    
    try:
        # Get configuration from environment
        experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'fraud-detection-production')
        max_runs = int(os.getenv('MAX_MLFLOW_RUNS_TO_KEEP', '5'))
        
        print(f"🔍 MLflow URI: {mlflow_uri}")
        print(f"🔍 Looking for experiment: {experiment_name}")
        print(f"📊 Max runs to keep: {max_runs}")
        
        # Get or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
        except Exception as e:
            print(f"⚠️ Error getting experiment: {e}")
            experiment = None
        
        if experiment:
            print(f"✅ Found experiment: {experiment.name} (ID: {experiment.experiment_id})")
            
            # Get all runs for this experiment
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"]  # Most recent first
            )
            
            print(f"📈 Found {len(runs)} existing runs")
            
            if len(runs) > max_runs:
                # Keep the most recent runs, delete the rest
                runs_to_keep = runs.iloc[:max_runs]
                runs_to_delete = runs.iloc[max_runs:]
                
                print(f"🗑️ Will delete {len(runs_to_delete)} old runs...")
                
                deleted_count = 0
                failed_count = 0
                
                for _, run in runs_to_delete.iterrows():
                    try:
                        run_id = run.run_id
                        run_name = run.get('tags.mlflow.runName', 'Unknown')
                        start_time = run.start_time
                        
                        # Delete the run
                        mlflow.delete_run(run_id)
                        print(f"🗑️ Deleted run: {run_name} ({run_id[:8]}...) from {start_time}")
                        deleted_count += 1
                        
                    except Exception as e:
                        print(f"⚠️ Could not delete run {run_id}: {e}")
                        failed_count += 1
                
                print(f"✅ Deleted {deleted_count} old runs")
                if failed_count > 0:
                    print(f"⚠️ Failed to delete {failed_count} runs")
            
            else:
                print(f"ℹ️ No cleanup needed. Current runs ({len(runs)}) <= max ({max_runs})")
            
            # Show remaining runs
            if len(runs) > 0:
                print(f"\n📋 Remaining runs:")
                remaining_runs = runs.iloc[:max_runs] if len(runs) > max_runs else runs
                for _, run in remaining_runs.iterrows():
                    run_name = run.get('tags.mlflow.runName', 'Unknown')
                    accuracy = run.get('metrics.accuracy', 'N/A')
                    start_time = run.start_time
                    print(f"  • {run_name} - Accuracy: {accuracy} - {start_time}")
            
            print(f"✅ MLflow cleanup completed. Keeping {min(len(runs), max_runs)} runs.")
            
        else:
            print("📝 Experiment not found, will be created during training.")
            
            # Create experiment if it doesn't exist
            try:
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"✅ Created new experiment: {experiment_name} (ID: {experiment_id})")
            except Exception as e:
                print(f"⚠️ Could not create experiment: {e}")
                
    except Exception as e:
        print(f"⚠️ MLflow cleanup error: {e}")
        # Don't exit with error for cleanup failures
        print("⚠️ Continuing despite cleanup error...")

def cleanup_old_artifacts():
    """Clean up old model artifacts from disk"""
    try:
        artifacts_dir = "./mlartifacts"
        if os.path.exists(artifacts_dir):
            # Get all subdirectories (experiments)
            for exp_dir in os.listdir(artifacts_dir):
                exp_path = os.path.join(artifacts_dir, exp_dir)
                if os.path.isdir(exp_path):
                    # Clean up artifacts older than 30 days
                    cutoff_time = datetime.now() - timedelta(days=30)
                    
                    for root, dirs, files in os.walk(exp_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                            
                            if file_time < cutoff_time:
                                try:
                                    os.remove(file_path)
                                    print(f"🧹 Cleaned up old artifact: {file_path}")
                                except Exception as e:
                                    print(f"⚠️ Could not delete {file_path}: {e}")
            
            print("✅ Artifact cleanup completed")
        else:
            print("ℹ️ No artifacts directory found")
            
    except Exception as e:
        print(f"⚠️ Artifact cleanup error: {e}")

def show_mlflow_info():
    """Show MLflow tracking information"""
    try:
        mlflow_uri = mlflow.get_tracking_uri()
        print(f"\n📊 MLflow Tracking Info:")
        print(f"URI: {mlflow_uri}")
        
        # List all experiments
        experiments = mlflow.search_experiments()
        print(f"Experiments ({len(experiments)}):")
        
        for exp in experiments:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            print(f"  • {exp.name} - {len(runs)} runs")
            
    except Exception as e:
        print(f"⚠️ Could not get MLflow info: {e}")

if __name__ == "__main__":
    print("🧹 Starting MLflow cleanup...")
    
    # Show current MLflow info
    show_mlflow_info()
    
    # Cleanup old runs
    cleanup_old_runs()
    
    # Cleanup old artifacts
    cleanup_old_artifacts()
    
    print("✅ MLflow cleanup completed")
