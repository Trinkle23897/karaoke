# 第三方来源与许可

项目自行编写的代码与文档采用 [MIT License](LICENSE)。本页记录第三方代码、配置、数据和模型的来源、版本、许可及署名要求；这些内容保留各自的许可。

## 随仓库分发的代码和配置

| 内容 | 来源与版本 | 许可与本地修改 |
| --- | --- | --- |
| `karaoke/web/vendor/signalsmith-stretch.mjs` | [Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch/tree/a670068d9aeb64913331d5cc29337b19a457a7df)，Geraint Luff / Signalsmith Audio Ltd. | [MIT 全文](karaoke/web/vendor/signalsmith-stretch.LICENSE.txt)。本地修改让原调跳过变调运算，修复声道数变化时的重新配置，并提供直接加载内嵌 WASM 的入口，供没有 AudioWorklet 的浏览器使用。原版和本地哈希见 [pitch-shift.json](karaoke/dependencies/pitch-shift.json)。 |
| `karaoke/benchmark_mimo.py` 的分块与交叉淡化实现 | [Sony MIMO 的 evaluate.py](https://github.com/SonyResearch/mimo-audio-separation/blob/9169bbbf9e6fdca5b22d5735b4244fbad7ab40b5/src/evaluate.py)，Yukara Ikemiya | 改编用于本项目的离线评测；保留 [MIT 全文](licenses/mimo-MIT.txt)。模型与上游源码不随仓库提供。 |
| `karaoke/models/pretrained/vocals_mel_band_roformer*.yaml` | [KimberleyJSN / Mel-Band RoFormer](https://huggingface.co/KimberleyJSN/melbandroformer) | 上游模型卡声明 MIT。保留上游配置及本项目的推理配置；权重哈希见 [separation.json](karaoke/dependencies/separation.json)。 |
| `karaoke/models/pretrained/karaoke_bs_roformer_anvuew.yaml` | [Anvuew 的固定版本](https://huggingface.co/anvuew/karaoke_bs_roformer/tree/0d4423d42e12cf2ba39ae09171028507b8a2a7be) | 该版本模型卡声明 `gpl-3.0`，保留 [GPL v3 全文](licenses/GPL-3.0.txt)。 |

## 另行安装的 Python 组件与数据

下列直接依赖由 `requirements*.txt` 安装，包本身不随源码仓库分发。制作包含依赖的发行包时，一并保留对应许可、署名及所需源码。

| 组件 | 固定版本、用途与许可 |
| --- | --- |
| [cmudict](https://pypi.org/project/cmudict/1.1.3/) | 1.1.3，GPL-3.0-or-later；`gaussian_align.py` 中的研究工具直接导入。包内 CMU 发音字典单独使用 BSD 类许可。 |
| [phonemizer](https://github.com/bootphon/phonemizer/tree/v3.3.0) | 3.3.0，GPL-3.0-or-later；英文歌声音素对齐使用其 eSpeak 接口。[上游许可](https://github.com/bootphon/phonemizer/blob/v3.3.0/LICENSE)。 |
| [pykakasi](https://pypi.org/project/pykakasi/2.3.0/) | 2.3.0，GPL-3.0-or-later；日文发音处理直接导入。 |
| [espeakng-loader](https://pypi.org/project/espeakng-loader/0.2.4/) / [eSpeak NG](https://github.com/espeak-ng/espeak-ng) | loader 0.2.4；loader 上游声明 MIT，所加载的 eSpeak NG 为 GPLv3 系列许可。该版本本机 wheel 含原生库及数据，打包前还需补齐可核验的二进制来源、对应源码与许可通知。 |
| [uroman](https://pypi.org/project/uroman/1.3.1.1/) | 1.3.1.1，MIT 类许可附额外项目署名要求；用于多语言声学复核的转写。该版本包分类写 Apache，实际条款以附带的许可文件为准，见下方。 |
| [PyCantonese](https://github.com/jacksonllee/pycantonese) | 5.0.0，Python 代码 MIT；粤拼函数使用 HKCanCor 与 rime-cantonese 数据，二者为 CC-BY-4.0。包内另有 CantoMap（GPLv3）、Common Voice（MPL-2.0）、CTCPC（CC0）数据；制作整包时保留各数据子目录的许可。 |
| [pyopenjtalk](https://github.com/r9y9/pyopenjtalk) | 0.4.1，Python 封装 MIT；本项目使用文本前端。包内 Mei voice 单独采用 CC-BY-3.0。 |

GPL 允许商业使用；分发受其覆盖的程序组合时，按适用条件提供源码、通知和许可。组合程序与独立作品集合的区别见 [GNU FAQ](https://www.gnu.org/licenses/gpl-faq.en.html#IfInterpreterIsGPL)，完整条款见 [GPL v3](licenses/GPL-3.0.txt)。

### uroman 署名

This project uses the universal romanizer software 'uroman' written by Ulf Hermjakob, USC Information Sciences Institute (2015-2020).

Ulf Hermjakob, Jonathan May, and Kevin Knight. 2018. *Out-of-the-box universal romanization tool uroman*. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, Demo Track.

固定版本随包的[完整许可](licenses/uroman-LICENSE.txt)保留原始版权年份与附加署名条款。

### 粤拼数据署名

- HKCanCor：K. K. Luke and May L. Y. Wong (2015). *The Hong Kong Cantonese Corpus: Design and Uses*. Journal of Chinese Linguistics Monograph Series, 25, 312–333。[数据来源](https://github.com/fcbond/hkcancor)，经 PyCantonese 使用，许可为 [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)。PyCantonese 调整了上游数据格式，本项目未再修改随包数据。
- rime-cantonese：rime-cantonese contributors 的[粤语拼音词库](https://github.com/rime/rime-cantonese/tree/33f7d81bc7ef5eb2f69538a2d3d5fc109529e2dd)与 [CanCLID 上游数据](https://github.com/CanCLID/rime-cantonese-upstream/tree/bea0d7a366627506eb214d82b80780b5db4b05a1)，经 PyCantonese 转为运行时词典，许可为 [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)。项目查询词典生成读音注释，不修改随包词典；文档示例说明见 [截图素材](docs/images/README.md)。

## 另行安装的对齐源码与模型

- audio-separator 0.44.5：调用入口的[上游许可为 MIT](https://github.com/nomadkaraoke/python-audio-separator/blob/v0.44.5/LICENSE)，模型许可按具体权重列出。
- SOFA：固定源码版本的[许可为 MIT](https://github.com/qiuqiao/SOFA/blob/9549c6a86d16019c817eefe4bb8183405da524cc/LICENSE)；其他作者提供的模型、字典和训练数据采用各自许可。
- LyricsAlignment-Multilingual：固定源码版本的[许可为 MIT](https://github.com/jhuang448/LyricsAlignment-Multilingual/blob/ca1a3923d6c8bf7d20eefa1080f3f104a6173d38/LICENSE)。来源与校验值见 [english-alignment.json](karaoke/dependencies/english-alignment.json)。
- MMS_FA：可选的声学复核模型，[许可为 CC-BY-NC 4.0（非商业）](https://docs.pytorch.org/audio/stable/generated/torchaudio.pipelines.MMS_FA.html)。商业部署需另行取得适用授权或替换模型。
- SOFA Japanese `akm_ja_v001`：[发布页](https://github.com/ariikamusic/SOFA_Models/releases/tag/akm_ja_v001)及所附 readme 未提供明确的权重许可，使用和分发权限需向权重发布者确认。其他作者提供的 SOFA 权重、字典和 G2P 资源也按各自来源核实。
- SenseVoice Small：官方源码为 MIT，权重采用独立的模型协议。[官方说明](https://github.com/QwenAudio/SenseVoice#license)允许按协议商业使用，要求署名并保留模型名称。本项目使用 sherpa-onnx 发布的 int8 转换包，保留其模型许可。
- Qwen3 ForcedAligner 0.6B：[固定版本模型卡](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B/blob/c7cbfc2048c462b0d63a45797104fc9db3ad62b7/README.md)声明 Apache-2.0；相关文件身份见依赖清单。
- Whisper 等其余资源按 [SETUP](SETUP.md) 从原发布者取得；[依赖清单](karaoke/dependencies)记录版本和哈希。
- 早期对照模型的来源可能没有完整许可声明，因此不随仓库提供其权重或配置，也不作为默认安装依赖。

Python 包由各 `requirements*.txt` 固定版本，Node.js、FFmpeg、LAME 和系统字体另行安装。制作容器、安装包或公开演示服务时，按实际包含的直接依赖、间接依赖和模型逐项核对许可。

## 开发检查工具

Ruff 和 ty 的版本固定在 [requirements-lint.txt](requirements-lint.txt)；ESLint、`@eslint/js` 和 globals 固定在 [package.json](package.json)，间接依赖由 [package-lock.json](package-lock.json) 锁定。这些工具仅用于开发和 CI，许可证见 [Ruff](https://github.com/astral-sh/ruff/blob/main/LICENSE)、[ty](https://github.com/astral-sh/ty/blob/main/LICENSE)、[ESLint](https://github.com/eslint/eslint/blob/main/LICENSE) 和 [globals](https://github.com/sindresorhus/globals/blob/main/license)。

开发环境另行安装 [Hydra](https://github.com/facebookresearch/hydra/blob/v1.3.2/LICENSE) 和 [OmegaConf](https://github.com/omry/omegaconf/blob/v2.3.0/LICENSE)，供 MIMO 研究工具及全仓类型检查使用，版本见 [requirements-dev.txt](requirements-dev.txt)。

## 音乐和在线来源

歌曲、MV、封面和歌词的使用与分发遵循内容权利人的授权；访问和下载同时遵循来源平台条款，例如 [YouTube 条款](https://www.youtube.com/static?template=terms)。README 截图的展示范围、出处和授权状态见[截图素材说明](docs/images/README.md)，数据与凭据的提交方式见[贡献指南](CONTRIBUTING.md#数据与凭据)。
