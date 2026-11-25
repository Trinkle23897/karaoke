# 参与开发

先按 [SETUP](SETUP.md) 准备环境。[源码导航](karaoke/README.md)区分服务入口、正式处理阶段和研究工具；项目约束见 [AGENTS.md](AGENTS.md)。

## 修改与验证

- 围绕具体问题提交小改动，说明触发条件、修改后的行为和验证结果。优先修复共同机制，避免歌曲特例。
- 扩展已有行为测试，覆盖实际故障和关键边界。
- 在独立、空的数据目录验证服务启动；测试使用隔离房间和合成数据，真实下载另行验证。
- UI 改动检查桌面、手机和全屏；音频／字幕改动遵循[验收标准](karaoke/ACCEPTANCE.md)，保留完整失败样本与测量结果。

安装开发工具并运行检查：

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
npm ci
.venv/bin/ruff check .
.venv/bin/ty check --python .venv/bin/python
npm run lint
```

Ruff 检查自有 Python 代码的错误、导入顺序和现代 Python 写法；ty 检查整个 `karaoke/`，包括算法、工具和测试；ESLint 检查前端 JavaScript、测试和自身配置。工具版本已固定，第三方源码和生成文件不纳入检查。类型问题优先补准数据边界；只有第三方声明不准确或必要的运行时替换，才在具体行注明原因并局部忽略。

完整回归命令：

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=karaoke .venv/bin/python -m pytest -q karaoke/test_api.py karaoke/test_room.py karaoke/test_algorithms.py
npm test
```

测试需要主环境依赖和 SOFA / pyopenjtalk 发音字典，不需要歌曲库、模型权重或 API key。CI 运行同一套 lint、类型检查、行为测试及空数据启动检查。

## 文档写法

先说明能做什么、怎么用；遇到限制，给出原因和可执行的下一步。许可详情集中在 [THIRD_PARTY.md](THIRD_PARTY.md)，质量指标集中在[验收文档](karaoke/ACCEPTANCE.md)，使用指南通过链接引用，减少重复的免责声明。性能数据注明硬件、版本、缓存条件和计时范围。

## 数据与凭据

歌曲、歌词库、模型权重、生成文件、数据库、凭据、个人部署信息和运行报告留在仓库外或 Git 已忽略的数据目录。测试优先使用合成音频与自写短文本。文档截图等需要提交的第三方素材，使用授权明确的内容，并在[素材说明](docs/images/README.md)记录来源、展示范围和许可。

提交前检查 `git diff --cached --stat` 和 `git diff --cached`。发布前对整个 Git 历史做密钥扫描，例如安装 Gitleaks 后运行：

```sh
gitleaks git . --log-opts=--all --redact=100
```

另行检查个人信息及 Git 历史；发现凭据泄露时先撤销／轮换，再与维护者安排历史清理。提交问题报告前去掉令牌、cookies、私人地址和完整歌词。

## 部署与许可

服务默认支持局域网访问，所有设备共享歌单。本机监听和公网访问控制的配置见 [SETUP](SETUP.md#5-运行配置)。

项目自行编写的代码与文档采用 [MIT License](LICENSE)。贡献内容应可按项目许可提供；新增依赖、引入的代码和模型配置保留原许可、出处及署名，并更新 [THIRD_PARTY.md](THIRD_PARTY.md)。
