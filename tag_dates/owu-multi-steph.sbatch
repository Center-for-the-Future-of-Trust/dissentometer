#! /bin/bash

#SBATCH -n 1 -c 18 --mem 32g --gres=gpu:1
#SBATCH -o openwebui-%j.log
#SBATCH --job-name "webui-multi"



##### Adjust the BASE directory and PORT_MAPs according to your computing environment
BASE="/local/scratch/${USER}/openwebui"
INSTANCES="/l00 /l01 /l02"
# Associative arrays for ports
declare -A WEBUI_PORT_MAP
declare -A OLLAMA_PORT_MAP
WEBUI_PORT_MAP["/l00"]=58111
OLLAMA_PORT_MAP["/l00"]=58112
WEBUI_PORT_MAP["/l01"]=58113
OLLAMA_PORT_MAP["/l01"]=58114
WEBUI_PORT_MAP["/l02"]=58115
OLLAMA_PORT_MAP["/l02"]=58116


#### Internal variables
WEBUI_VER="v0.6.26"
CT_NAME="sgowui"
CT_IMAGE="openwebui-ollama-${WEBUI_VER}.sif"
CT_OVRLAY="openwebui-overlay.sif"
CT_OVRLAY_SIZE=10000         # In MiB
CP_PORT=8080
SERVER="$(hostname)"

MODELS="llama3.1:8b gpt-oss:20b"
# Ollama configuration variables
OLLAMA_NUM_PARALLEL=64 		# Defaults to 1 or 4


cd ${BASE}

for INSTANCE in ${INSTANCES}; do

        mkdir -p ${BASE}${INSTANCE}
        mkdir -p ${BASE}${INSTANCE}/ollama ${BASE}${INSTANCE}/open-webui/work

        WEBUI_PORT=${WEBUI_PORT_MAP[$INSTANCE]}
        OLLAMA_PORT=${OLLAMA_PORT_MAP[$INSTANCE]}

        echo "Starting OpenWEBUI instance ${INSTANCE} with WEBUI_PORT=${WEBUI_PORT} and OLLAMA_PORT=${OLLAMA_PORT}"

        # Downloading image as apptainer image
        # Images are shared by instances
        if [ ! -f ${BASE}/${CT_IMAGE} ]; then
          echo "Downloading OpenWEBUI container image. It might take some time ..."
          apptainer pull ${BASE}/${CT_IMAGE} docker://ghcr.io/open-webui/open-webui:${WEBUI_VER}-ollama
        fi

        # Creating overlay image to allow RW filesystem. Size in MiB
        if [ ! -f ${BASE}${INSTANCE}/${CT_OVRLAY} ]; then
          echo "Creating overlay image ..."
          apptainer overlay create --size ${CT_OVRLAY_SIZE} ${BASE}${INSTANCE}/${CT_OVRLAY}
        fi

        # Logging GPU IDs available
        echo "GPU IDs available are: ${CUDA_VISIBLE_DEVICES}"

        # Running the container
        ARGS="  --pid \
                --nv \
                --ipc \
                --no-home \
                -B ${BASE}${INSTANCE}/ollama:/root/.ollama \
                -B ${BASE}${INSTANCE}/open-webui/work:/home/${USER} \
                -B ${BASE}${INSTANCE}/open-webui:/app/backend/data \
                -B ${BASE}${INSTANCE}/ollama:$HOME/.ollama \
                -B ${BASE}${INSTANCE}/open-webui/work:$HOME \
                --env WEBUI_URL=http://localhost:${CP_PORT} \
                --env PORT=${WEBUI_PORT} \
                --env OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT} \
                --env OLLAMA_BASE_URL=http://${SERVER}:${OLLAMA_PORT} \
                --env OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL} \
                -o ${BASE}${INSTANCE}/${CT_OVRLAY} \
                ${BASE}/${CT_IMAGE} "

        apptainer run \
                ${ARGS} \
                bash -c "set -x; df -h; cd /app/backend; ./start.sh & sleep 5; for M in ${MODELS}; do ollama pull \\\$M; done; touch /tmp/tst; tail -f /tmp/tst" &> ${BASE}${INSTANCE}/out-${SLURM_JOB_ID}.log &

done

touch ${BASE}/running-${SLURM_JOB_ID}
tail -f ${BASE}/running-${SLURM_JOB_ID}
rm ${BASE}/running-${SLURM_JOB_ID}
