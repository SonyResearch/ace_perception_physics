
import subprocess
import os
import time


def launch_process(cmd):
    try:
        process = subprocess.Popen(
        cmd,shell=True
        )
    except Exception as e:
        print(f"Failed to start process: {e}")
        process = None
    return process



def list_files(path, extension):
    """TODO"""
    file_list = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(extension):
                file_list.append(os.path.join(root, file))

    return file_list

def main(path):
    paths=[os.path.dirname(p) for p in list_files(path,"tyo01.yaml")]

    for p in paths:
        for cluster in ["_a","_b"]:
            cmd=f"'/media/saraijiy/Data/ws/src/project_ace/install/aps/lib/aps/aps_video_publisher' --path '{p}' --cluster {cluster}"
            print("Processing " + p+ " with cluster " + cluster)
            print(cmd)
            process=launch_process(cmd)
            retcode=-1
            if process is not None:
                try:
                    process.wait()
                    retcode=process.returncode
                finally:
                    process.poll()  # Clean up zombie processes

                # Kill processes containing 'player' or 'racket'
                try:
                    subprocess.run("pkill -f 'player|racket'", shell=True, check=False)
                except Exception as e:
                    print(f"Failed to kill processes: {e}")

            # if retcode ==0:
            # input("Process completed successfully. Press Enter to continue...")
main("/mnt/tokyo_nas/shared_nas/univ_players_recordings/20240510/")
