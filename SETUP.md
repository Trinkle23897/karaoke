# 安装与运行

所有命令从仓库根目录执行。按下面的步骤准备服务、处理环境和模型，再完成第一首歌。

## 1. 网页与任务服务

只启动界面不需要下载模型：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r karaoke/requirements-api.txt
.venv/bin/python -m karaoke.api
```

在电脑上打开 <http://127.0.0.1:8746/>。手机或平板连上同一 Wi-Fi 后，打开 `http://电脑的局域网IP:8746/`。初次启动是空曲库，运行数据保存在 Git 已忽略的 `karaoke/service-data/`。

## 2. 完整处理环境

### 安装前检查内存

单首处理推荐 Apple Silicon Mac 配备 24 GB 统一内存，NVIDIA 主机配备 32 GB 系统内存和 16 GB 显存。浏览器端不需要加载模型，可以连接另一台处理主机。

macOS 用 `sysctl -n hw.memsize` 查看物理内存（字节），用「活动监视器 → 内存」观察内存压力和交换空间；Linux 可用 `free -h`。磁盘上默认为歌曲缓存设置 10 GiB 预算，另为模型、依赖和运行时临时文件留出空间。

按上述配置运行时，保持单首准备、单个计算工位和最小批次：

```sh
export KARAOKE_PREPARATION_CONCURRENCY=1
export KARAOKE_COMPUTE_CONCURRENCY=1
export KARAOKE_SEPARATION_BATCH_SIZE=1
export KARAOKE_WHISPER_BATCH_SIZE=1  # 仅本地 Whisper 模式使用
```

前三项默认 1；本地 Whisper 批次默认 8，内存预算有限时改为 1。同时准备多首歌或调大批次需要额外内存和显存。

需要 `ffmpeg`、`ffprobe` 和 Node.js 22+ 在 PATH 中；严格音频验收及独立 MP3 导出还需要 `lame`。例如 macOS 使用 `brew install ffmpeg lame node`，Debian/Ubuntu 使用 `sudo apt-get install ffmpeg lame libsndfile1` 并安装 Node.js 22+。普通点歌只生成底轨（伴奏＋伴唱）与主唱两轨 FLAC，不调用 LAME；FFmpeg 负责母版解码、公共增益转换与视频编码。YouTube 下载启用 Node.js 执行播放器脚本；`yt-dlp[default]` 同时安装与下载器版本配套的 EJS 组件，不在下载时拉取额外脚本。Node.js 也用于播放器和音频变调测试，前端本身不需要构建。实时升降调的 Signalsmith Stretch / WASM 随源码提供，不需要额外安装、模型下载或运行时 CDN；版本和许可见 `karaoke/dependencies/pitch-shift.json`。

更新代码后也要重新安装 `karaoke/requirements-api.txt`，使运行环境使用项目固定的下载器版本。旧版 yt-dlp 可能因 YouTube 播放器更新而出现 `Signature solving failed`、`n challenge solving failed`，甚至 `a.addOnDisposeCallback is not a function`；遇到这些诊断时先核对 `.venv/bin/python -m yt_dlp --version` 与依赖文件，安装匹配版本后重试原任务。不要仅凭这类错误更换歌曲链接或重新导出 cookies。

主环境负责音频导出、ASR、歌词整理、对齐及字幕渲染。先按硬件安装匹配版本的 PyTorch 与 torchaudio，再装项目依赖。当前 Mac / MPS 开发环境使用 Python 3.12、`torch==2.11.0`、`torchaudio==2.11.0`：

```sh
.venv/bin/python -m pip install torch==2.11.0 torchaudio==2.11.0
.venv/bin/python -m pip install -r requirements-dev.txt
```

CUDA 机器选择与驱动兼容的 PyTorch CUDA wheel，并保持 torch/torchaudio 版本匹配。CPU 也能执行，整曲处理更适合使用 GPU。

分离使用独立环境，避免 audio-separator 的依赖影响主环境：

```sh
python3.12 -m venv .venv-separator
.venv-separator/bin/python -m pip install 'audio-separator[cpu]==0.44.5' 'librosa==0.11.0'
.venv-separator/bin/python -c 'from audio_separator.separator import Separator'
export KARAOKE_SEPARATOR_PYTHON="$PWD/.venv-separator/bin/python"
```

该环境也需安装适合硬件的 PyTorch。`cpu` extra 补齐导入时必需的 ONNX Runtime；Roformer 分离通过 PyTorch 使用 CUDA / MPS。固定 librosa 0.11.0 是为了提供分离器需要的 audioread。安装后运行上面的导入命令，确认入口及其依赖可用。

已验证的 Linux 组合为 Ubuntu 24.04、RTX 5090、驱动 580.126.09，以及主环境和分离环境的 `torch==2.11.0` / `torchaudio==2.11.0`（CUDA 13.0 wheel）。Mac 分离环境另使用 torch 2.13.0。`benchmark_neural.py` 会检查 audio-separator 版本，并按模型配置设置类型、重叠和推理参数；权重使用下表中的配套版本。

## 3. 对齐器源码

### 中文／日文：SOFA

```sh
mkdir -p karaoke/research/subtitles-2026-09-18
git clone --filter=blob:none --no-checkout https://github.com/qiuqiao/SOFA.git karaoke/research/subtitles-2026-09-18/sofa
git -C karaoke/research/subtitles-2026-09-18/sofa checkout --detach 9549c6a86d16019c817eefe4bb8183405da524cc
```

中文发音测试需要该源码中的 `dictionary/opencpop-extension.txt`，无需模型权重。日文使用 pyopenjtalk，首次调用会按需下载发音字典。SOFA 在这里提供源码与字典，Python 依赖统一使用本项目的 `requirements-full.txt`。

### 英文：LyricsAlignment-Multilingual

```sh
mkdir -p karaoke/research/word-onset-2026-09-18
git clone --filter=blob:none --no-checkout https://github.com/jhuang448/LyricsAlignment-Multilingual.git karaoke/research/word-onset-2026-09-18/lyrics-alignment-multilingual
git -C karaoke/research/word-onset-2026-09-18/lyrics-alignment-multilingual checkout --detach ca1a3923d6c8bf7d20eefa1080f3f104a6173d38
cp karaoke/dependencies/english-alignment.json karaoke/research/word-onset-2026-09-18/lyrics-alignment-multilingual/source-manifest.json
```

需要其中的 `model.py`、`utils.py` 和 `checkpoints/checkpoint_Baseline`。源码和 checkpoint 的 SHA-256 见 [english-alignment.json](karaoke/dependencies/english-alignment.json)。该 checkout 已包含 checkpoint，整个第三方目录由 Git 忽略；Python 依赖沿用本项目主环境。

## 4. 模型权重

按下表下载原始权重并放到对应位置，使用仓库中配套的 YAML 配置。

| 用途 | 下载来源 | 本地位置 |
|---|---|---|
| F：音乐／人声分离 | [Mel-Band RoFormer](https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt) | `karaoke/models/pretrained/vocals_mel_band_roformer.ckpt` |
| 主唱／伴唱分离 | [Anvuew 固定版本](https://huggingface.co/anvuew/karaoke_bs_roformer/resolve/0d4423d42e12cf2ba39ae09171028507b8a2a7be/karaoke_bs_roformer_anvuew.ckpt) | `karaoke/models/pretrained/karaoke_bs_roformer_anvuew.ckpt` |
| 中文音素对齐 | [SOFA Mandarin](https://github.com/qiuqiao/SOFA/releases/download/v1.0.1/pretrained_mandarin_singing.zip) | 解压到 `karaoke/models/alignment/sofa/mandarin/`，保留压缩包子目录 |
| 日文音素对齐 | [SOFA Japanese](https://github.com/ariikamusic/SOFA_Models/releases/download/akm_ja_v001/akm_ja_v001.zip) | 解压到 `karaoke/models/alignment/sofa/japanese/`，保留压缩包子目录 |
| 英文音素对齐 | 上一步的固定版本仓库 | `karaoke/research/word-onset-2026-09-18/lyrics-alignment-multilingual/checkpoints/checkpoint_Baseline` |
| 中日文歌曲中英文词组的复核（可选） | [SOFA English tgm_en_v100](https://github.com/spicytigermeat/LabelMakr/releases/download/tgm_en_v100/tgm_en_v100.zip) | 解压到 `karaoke/models/alignment/sofa/english/`，保留 `tgm_en_v100/` 子目录及其字典、G2P 文件 |
| 粤语语言分类 | [SenseVoice Small int8](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2) | 解压后将 `model.int8.onnx`、`tokens.txt` 和 `LICENSE` 放到 `karaoke/models/alignment/sensevoice/`，校验值见 [sensevoice.json](karaoke/dependencies/sensevoice.json) |
| Whisper 粗对齐 | stable-whisper 自动下载 | `karaoke/models/alignment/whisper/` |
| 可选本地批量 ASR | `Systran/faster-whisper-large-v3`（完整 CTranslate2 模型） | `KARAOKE_WHISPER_MODEL` |
| 合并字组及中日文提前句首的独立声学复核（可选） | [MMS_FA](https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt) | `karaoke/models/alignment/mms/model.pt` |
| 中文残余字组的后备复核（可选） | [Qwen3 ForcedAligner 0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B/tree/c7cbfc2048c462b0d63a45797104fc9db3ad62b7) | `karaoke/models/alignment/qwen3-forced-aligner-0.6b/` |

权重和压缩包的校验值分别记录在 [separation.json](karaoke/dependencies/separation.json)、[sofa-mandarin.json](karaoke/dependencies/sofa-mandarin.json)、[sofa-japanese.json](karaoke/dependencies/sofa-japanese.json)。`separation.json` 也保留了早期对照模型的来源；正式流程只需要 F 和 Anvuew。可用 `shasum -a 256 文件路径` 核对下载内容。上游若更新了 `main` 对应的文件，应以这里记录的 SHA-256 判断是否还是同一个模型。

模型许可、商业使用条件和署名要求见[第三方说明](THIRD_PARTY.md)，其中列明 Anvuew 的 GPL 声明与可选 MMS_FA 的非商业条款。

MMS_FA 的校验值见 [mms-alignment.json](karaoke/dependencies/mms-alignment.json)。它复核仍然合并的中文字组、中日文歌曲中纯英文句的合并词组，以及疑似提前过久的中日文句首：分别对原音和分离人声对齐，再与 SOFA 的三个上下文结果比较。运行中不自动下载该模型；缺少权重会保留原结果与原因，不能把未能拆开的字词标成通过。没有问题的字词不调用它。

英文词组复核还需要 LabelMakr 的 G2P 源文件，放到
`karaoke/research/word-onset-2026-09-18/labelmakr/models/g2p_model.py`。
固定源码 URL、模型与字典哈希见 [sofa-english.json](karaoke/dependencies/sofa-english.json)；
该模块复用主环境的 PyTorch，不需要安装 LabelMakr GUI 或覆盖主环境依赖。

后备 Qwen 复核在独立子进程运行，避免其 Transformers 版本与主环境冲突。按上面固定版本下载完整模型目录，文件哈希见 [qwen-forced-aligner.json](karaoke/dependencies/qwen-forced-aligner.json)。可创建独立环境，安装匹配本机的 torch 及 `karaoke/requirements-qwen.txt`，然后设置 `KARAOKE_QWEN_PYTHON=/绝对路径/.venv-qwen/bin/python`。已有的 `karaoke/research/subtitles-2026-09-18/python-deps` 也只在这个子进程加载。该路径仅处理此前仍未拆开的中文字组，禁用 SDK 时间插值；没有依赖或证据不符时保留原失败，不自动下载或改写时间。

## 5. 运行配置

通过本机环境变量设置 `OPENAI_API_KEY`，凭据留在本机配置中。普通点歌优先使用已发表歌词；元数据不足时调用文本模型补充，默认 `gpt-5-mini`。严格音频验收另行使用 ASR，默认在线调用。

可选本地识别：安装 `karaoke/requirements-local-asr.txt`，下载上述完整模型目录，
设置 `KARAOKE_ASR_BACKEND=local`、`KARAOKE_WHISPER_MODEL=/绝对路径/模型目录`。
本地模式无需 API key；支持 CUDA 或 CPU，CUDA 需要 CTranslate2 对应的 CUDA 12 cuBLAS 和 cuDNN 9。
模型固定版本为 `edaa852ec7e145841d8ffdb056a99866b5f0a478`，下载后保存文件 SHA-256。
`KARAOKE_WHISPER_BATCH_SIZE` 默认 8；先在目标硬件验证批次对照和完整歌曲，再用于回归验收。

```sh
export KARAOKE_SEPARATOR_PYTHON="$PWD/.venv-separator/bin/python"
export KARAOKE_DEVICE=auto
.venv/bin/python -m karaoke.api
```

`auto` 按 CUDA → MPS → CPU 选择设备。Linux 还需用 `KARAOKE_FONT_FILE` 指定覆盖目标语言的字体文件，例如已安装的 Noto Serif CJK 字体；遇到缺字提示时更换字体后重试。完整环境变量见 [API.md](karaoke/API.md)。

每个数据目录运行一个 Uvicorn 服务进程。`KARAOKE_PREPARATION_CONCURRENCY=none` 允许所有已点歌曲进入阶段流水，默认值为 `1`。下载、分离、识别／对齐、音轨封装和视频编码各有共享工位。不同工位可以交叠，等待工位的任务优先处理当前待唱歌曲，再按歌单顺序处理，不中断已有步骤。任务共用缓存预算，各自保存进度、结果和错误。

服务默认监听所有 IPv4 接口（`0.0.0.0:8746`）。能连到这个端口的人都能点歌和管理共享歌单，适合在可信的家庭网络里使用。播放时点底栏的 `♭` 或 `♯` 就能升降调，无需域名或证书。

只在本机使用，或只通过本机的 Cloudflare Tunnel 对外提供服务时，启动前运行 `export KARAOKE_HOST=127.0.0.1`。使用隧道时，将 HTTPS 子域名指向 `http://127.0.0.1:8746`，用 Cloudflare Access 等访问控制保护整个域名及其 API，并验证匿名请求受限、登录后媒体和 SSE 正常。

长期在本机运行时，将服务与临时终端分开，并保存日志（macOS / Linux）：

```sh
mkdir -p karaoke/service-data
nohup .venv/bin/python -u -m karaoke.api </dev/null >>karaoke/service-data/server.log 2>&1 &
```

环境变量需在启动前设置。检查 `curl -fsS http://127.0.0.1:8746/health` 和 `tail -n 50 karaoke/service-data/server.log`；启动终端退出后应再次检查健康接口。`nohup` 不负责崩溃重启或开机启动；需要这两项时交给操作系统的服务管理器。重启前确认没有正在准备或播放的歌曲，再向监听进程发送 `TERM`，等它退出后启动，避免两个 worker 抢同一数据目录。

新歌一直排队、`running_job` 为空时，检查 `/health` 的 `preparation_paused`。定向回归使用的 `KARAOKE_DATA_DIR/preparation.pause` 会暂停普通点歌处理；恢复日常使用前，确认同一数据目录没有定向处理进程，再移除这个暂停文件。回归清单旁的 `audio.pause` 和 `manifest.pause` 可继续保留，不影响普通点歌。

若网页能打开但下载失败，且日志出现 `Unable to connect to proxy` / `Tunnel connection failed: 403 Forbidden`，检查服务进程继承的代理环境。用 `python -m yt_dlp --ignore-config --skip-download --print title 视频URL` 在同一环境验证下载入口。通过编程助手启动长期服务时，沿用本机已验证的视频下载网络配置；工具临时注入的代理可能需要在服务启动配置中调整。

## 6. 准备第一首歌

安装完成后，完整准备并播放第一首歌，把下载、模型和浏览器一起验证：

1. 选择一首完整歌曲，例如周杰伦《稻香》，确认歌手和录音版本。通过网页搜索并点歌，或向 `POST /v1/room/entries` 提交 `{"url":"完整视频链接"}`。
2. 保存待唱项 `id` 和 `job_id`。查看原任务 `GET /v1/jobs/{job_id}`，等到 `status=completed`，并确认 `GET /v1/room` 中该待唱项可播放。
3. 按任务清单核对音轨、字幕、完整 MV 的大小和 SHA-256，完整解码音视频，检查字幕正文、时间范围与顺序。可使用 `karaoke.tools.soak.verify_outputs(..., playback_policy=True)`。
4. 在实际浏览器验证加载、定位、字幕显示和主唱开关。由编程助手操作时先用静音预览，完成后由使用者开唱。
5. 记录网页地址、歌名和来源、任务 ID、验证结果，以及启动／停止方法。保留可唱成品，清理已完成的隔离测试中间文件；遇到登录、下载或资源问题时，按错误提示处理后重试原任务。

选歌时可用「有歌词优先」找到已经查到正文的版本。「歌词未确认」的版本仍可点歌，准备流程会继续查询其他来源。需要隔离排查时，使用 [ProbeSession / run_probe](karaoke/API.md#先验证音频) 共用阶段锁，再由正常 worker 完成原任务。

## 7. 测试与范围

安装开发依赖、主环境 PyTorch 和 SOFA 源码后：

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=karaoke .venv/bin/python -m pytest -q karaoke/test_api.py karaoke/test_room.py karaoke/test_algorithms.py
node --test karaoke/test_player.mjs karaoke/test_pitch.mjs karaoke/test_search.mjs
```

这些行为测试不发起在线模型调用，不要求歌曲库或模型权重。真正验证下载、分离、对齐和渲染，仍需完成以上资源准备并提交一个真实视频链接。首次模型／字典下载耗时与后续单曲处理耗时应分别记录。

仓库的 [Tests workflow](.github/workflows/tests.yml) 在 push、PR 和手动触发时并行运行四个独立任务：Python Ruff、Python ty、Python 行为测试，以及 JavaScript lint 与行为测试，命令见[贡献指南](CONTRIBUTING.md#修改与验证)。Python 使用 3.12，JavaScript 使用 Node.js 22。ty 安装 Python 依赖后直接检查；Python 行为测试在 Ubuntu 24.04 上使用 CPU 版 PyTorch／torchaudio 2.11.0、上述固定版本的 SOFA 字典及 Noto Serif CJK 字体，准备日文发音字典后检查空库启动及全部行为测试。两组 JUnit 测试报告、启动日志和测试子进程日志保留 14 天。CI 不需要配置密钥；真实歌曲的分离质量和逐字起音验收仍单独进行。

正式入口是 `python -m karaoke.api`。`benchmark_neural.py` 和 `experiments/cjk-onsets-v003/run.py` 也是正式处理链的一部分；其他研究工具所需的素材和可选模型见[源码导航](karaoke/README.md)。
