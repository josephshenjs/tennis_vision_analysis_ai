# Rally Analysis：便携式 CUDA 网球回合分析

## Automatic workflow API update (2026-11-05)

`POST /analyses` also accepts optional `request_id` (64 lowercase hexadecimal
characters). Repeated requests with the same identity/key reuse the existing job;
a changed input or calibration returns 409. Each saved job includes its own court
snapshot. Unfinished jobs resume on process startup from Firestore.

The API now uses two bounded CPU job workers and one shared resident CUDA executor.
TrackNet and the racket models are loaded once per API process, on demand. Run one
API process per GPU. H.264 analysis video retains the input audio, and completed
job status includes `output_sha256` and `output_size` for download verification.
Configure `TENNIS_FFMPEG` if FFmpeg is not on PATH. GET /health reports API version 2.

## FastAPI: court calibration and R2 analysis

The API loads `.env.local` from this folder (existing environment variables take
precedence). Start it with:

```powershell
.\rally_analysis.cmd serve --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive request forms and the OpenAPI
schema. `GET /health` is an unauthenticated process health check.

For a fresh bundle, install the additional dependencies using a Python 3.11
installation with pip; the embedded runtime itself does not include pip:

```powershell
python -m pip install --target runtime/Lib/site-packages -r requirements-api.txt
python -m pip install --target runtime/Lib/site-packages --upgrade "typing-extensions==4.16.0"
```

Configuration is documented in [.env.example](.env.example). Firestore uses
`RALLYCLIP_FIREBASE_PROJECT` and the exact `RALLYCLIP_FIRESTORE_DATABASE` name.
Google authentication also requires Application Default Credentials, for example
`GOOGLE_APPLICATION_CREDENTIALS=C:/secrets/service-account.json` in `.env.local`.
Project/database names and R2 keys alone do not authenticate to Firestore.
The Google identity needs Firestore read/write access. R2 credentials need object
read/write access to `RALLYCLIP_ARTIFACT_BUCKET`.

Authenticate with `Authorization: Bearer <Firebase ID token>` when
`RALLYCLIP_REQUIRE_USER_AUTH=1`. Courts and jobs are scoped to that Firebase user.
Trusted backend callers may instead use `RALLYCLIP_ANALYSIS_API_TOKEN` or
`RALLYCLIP_API_TOKEN` as a Bearer token or `X-API-Token` header; these tokens bypass
Firebase user authentication and share a separate service court namespace.
Use the same authentication identity to save a court and submit its analysis.
Keep backend tokens on the server. If user authentication is disabled and neither
API token is configured, endpoints allow unauthenticated local use.

1. `PUT /courts/{court_id}` saves exactly four outer doubles-court baseline corners
   in original image pixels, ordered **far-left, far-right, near-right, near-left**.
   Include the original video resolution; analysis rejects a different resolution.

   ```json
   {
     "image_width": 1920,
     "image_height": 1080,
     "corners_px": [[700, 300], [1200, 300], [1700, 900], [200, 900]]
   }
   ```

   `GET /courts/{court_id}` retrieves the saved calibration. These example points
   must be replaced with points selected on the user's own video. Invalid,
   out-of-bounds, crossed or degenerate corners return HTTP 422.
   Firestore collection: `<RALLYCLIP_FIRESTORE_PREFIX>_courts`. Coordinates are
   stored as maps, and derived calibration as JSON, because Firestore does not
   support nested arrays.

2. `POST /analyses` accepts the saved court and an R2 video URL:

   ```json
   {
     "court_id": "camera1",
     "file_url": "https://ACCOUNT_ID.r2.cloudflarestorage.com/BUCKET/clips/point_0003.mp4"
   }
   ```

   The URL must be under `RALLYCLIP_R2_ENDPOINT/<RALLYCLIP_ARTIFACT_BUCKET>/`.
   For a public/custom R2 domain, configure `RALLYCLIP_R2_PUBLIC_BASE_URL` to the
   URL corresponding to the root of the same bucket. URL-encoded object names and
   presigned URL query strings are accepted; downloads use the configured R2
   credentials. Arbitrary external URLs are rejected.

   HTTP 202 returns `job_id`, `state: queued`, and `status_url`.
   The server downloads the input, runs the existing CUDA analysis pipeline,
   and uploads `analysis.mp4` to `analyzed/<job_id>/analysis.mp4` in the same bucket.

3. `GET /analyses/{job_id}` returns `queued`, `downloading`, `analyzing`,
   `uploading`, `complete`, or `failed`. On completion it includes `output_bucket`,
   `output_key`, and a fresh `output_url` valid for one hour. Failed jobs include
   `failed_stage` and a sanitized error. Job status is stored in
   `<RALLYCLIP_FIRESTORE_PREFIX>_analyses` in the configured Firestore database.

Run one API process/worker per GPU. Up to eight jobs (including the active job)
are accepted; further submissions return HTTP 429. Each job uses the bundled
Python/CUDA pipeline, and temporary local files are removed after processing.
Graceful shutdown waits for accepted jobs. Unfinished API jobs with a saved court snapshot resume from Firestore after a forced stop. Legacy jobs without a snapshot require resubmission. Model weights and NVIDIA/CUDA
requirements are the same as CLI analysis.

`run.py self-test` includes API tests with fake Firestore/R2 adapters; these test
validation, authentication, ownership, job completion, queue limits, and failures
without writing cloud data. They do not establish live cloud access.

Implementation references: [FastAPI background work](https://fastapi.tiangolo.com/tutorial/background-tasks/),
[Firestore Python client](https://docs.cloud.google.com/python/docs/reference/firestore/latest),
[Boto3 uploads](https://docs.aws.amazon.com/boto3/latest/guide/s3-uploading-files.html),
and [presigned download URLs](https://docs.aws.amazon.com/boto3/latest/reference/services/s3/client/generate_presigned_url.html).

局部高分辨率补检默认启用：双窗口融合后，对有可靠前序轨迹、最长12帧（约0.4秒）的缺失段，用原始图像384x216局部区域重新运行同一个常驻CUDA TrackNet。每段视频最多4次，优先较长缺失段。只推理缺失段及邻近上下文，仍需额外采样背景/解码；并非零开销。--no-local-recovery 可关闭。

补检须通过场地范围、至少2帧已有观测位置吻合、相邻检测支持及运动连续性检查；不覆盖已有观测，不用预测坐标填充。recovery.json 记录耗时、裁切范围、恢复帧和失败原因；fusion_audit.csv 的 local 表示补检实际观测。随后重新计算候选并进行球拍复核。

验证 outputs/local_recovery_validation：point_0013 找回7帧，91/160有支持候选保留，新增54待确认；point_0021 未恢复183–192，181仍不能评分。多尺度局部重试也未找回该段，当前权重仍有局限，不能把补检功能等同于该事件已解决。补检本身在这两段分别约2.8/3.2秒（本机测量，非性能保证）。43项测试中42通过、1项参考对照跳过。

PowerShell 流程现在直接调用本地 runtime/python.exe，避免 Ctrl+C 后出现 CMD 的 “Terminate batch job (Y/N)?”。直接运行 .cmd 仍可能出现 Windows 批处理确认提示。取消时 status.json 记录 cancelled；后台任务在检查点停止，已保存的标定和已完成结果保留。正在执行的原生 CUDA/模型初始化调用需要返回后才能停止，不保证即时中断。首次加载球拍模型会分别显示 detector / pose 初始化和 ready 提示。

低分补充候选路径默认启用：CatBoost 分数在 0.40 到当前高分阈值之间（默认 0.55）的峰值，只有前后各3个模型帧连续观测、没有缺失补点、出现下降转上升且水平运动连贯、没有异常跳点时，才进入同一球拍证据复核。相邻高分事件优先，避免同一次反弹重复记录。高分路径和模型原始评分不变。

events.json / candidates.json 的 candidate_path 区分 catboost_high_score 和 trajectory_supplement；supplemental_evidence 保存补充依据。补充候选并不自动确认为落地；仍可能被复核为击球或证据不足。新增路径额外需要3个模型帧的后续观测，连续漏检的事件不会由它补齐。

验证输出 outputs/supplement_validation/point_0013/analysis.mp4：第91帧（原始分数0.5302）新增为有支持候选，第160帧保留；point_0021 未新增候选。测试共38项，37通过，1项外部参考对照跳过。

## 启动

Windows x64 + NVIDIA 显卡驱动。无需激活conda/venv、无需另外安装Python，也不读取父目录中的模型或库。
保留整个 rally_analysis_ai 文件夹即可搬移。使用同一个入口 rally_analysis.cmd：

    .\rally_analysis_ai\rally_analysis.cmd analyze "C:\videos\point_0003.mp4" --court-calibration ".\rally_analysis_ai\calibrations\home_court.json" --far-roi 650 250 650 320

也可进入本文件夹后运行：

    .\rally_analysis.cmd analyze "C:\videos\point_0003.mp4" --far-roi 650 250 650 320

默认标定是本文件夹calibrations/home_court.json，默认输出本文件夹outputs/<视频名>/。
示例far-roi仅适用于当前home_court机位；换相机需重新标定并检查远端窗口。

相同入口的其他子命令：

    .\rally_analysis.cmd calibrate --video "C:\videos\tennis.mp4" --camera-id camera1
    .\rally_analysis.cmd analyze --help
    .\rally_analysis.cmd self-test

标定依次点击远左、远右、近右、近左四个双打外角，S保存、R全部重选、Q取消；没有单点拖动校正。
新标定位置calibrations/camera1.json，分析时用 --court-calibration 指定。

## 完整流程

原始视频 → 同时生成整场/远端两个窗口的TrackNet输入 → CUDA批次追踪 → 原图坐标融合
→ CatBoost候选 → 候选附近RacketVision RTMDet+RTMPose球拍证据 → 轨迹/接触复核
→ 一次编码最终视频，并保存CSV和JSON。

默认双窗口。--single-window 可关闭局部窗口。未给 --far-roi 时使用标定中的
far_tracking_roi_xywh；该字段不存在时由远半场地面几何加余量生成初始窗口。自动窗口不是
已验证的飞行范围，需确认不会裁掉高球、跑动和击球区。原图坐标始终保持一致。

复核保留全部原始候选，分组为 supported_bounce_candidate、likely_racket_hit、uncertain_*。
这些是证据分级，不是确认真值。缺少球拍不能当作没有球拍；166在本片仍是待确认而非已证明击球。
不包含IN/OUT。空中球使用带飞行高度余量的场地走廊，不直接限制在地面四边形内。

## 异步批量

    .\rally_analysis.cmd analyze "C:\videos\clip1.mp4" "C:\videos\clip2.mp4" --court-calibration calibrations/home_court.json --jobs 2 --batch-size 4
    .\rally_analysis.cmd analyze "C:\videos\same_camera_folder" --court-calibration calibrations/home_court.json --jobs 2

目录输入读取该目录顶层mp4/mov/mkv/avi。一次批量使用同一个标定，不同机位应分批运行。
同名视频会附加路径哈希区分输出。输入顺序和每段帧号都保留，失败任务不会取消其他视频。
失败时进程返回非零，错误写入对应error.log。status.json记录阶段，整个批次写summary.json。
Ctrl+C会停止调度和处理中帧循环，已完成结果保留。

## 性能机制

- 所有CUDA操作由同一个常驻线程持有，TrackNet、RTMDet、RTMPose在整个批次内各加载一次。
- 整场和远端合并为同一批模型输入，不再分别启动程序/重复加载模型。
- 顺序解码、CPU预处理和GPU推理通过有界队列重叠；不会丢帧或预读整个视频。
- 只对候选前后6帧运行球拍模型，使用原生像素上下文窗口。
- asyncio管理每视频任务；CPU候选/落盘/编码可与下一段GPU工作重叠。不是多个视频同时抢占CUDA。
- 只输出一份最终视频，避免旧流程多次编码的耗时和画质损失。

默认 --jobs 2、--cpu-workers 2、--prefetch 2、--batch-size 4。
batch-size表示源视频帧数，双窗口实际有2倍模型输入。显存不足时先降到2，再降到1。
增加jobs只增加在途任务，不会增加GPU推理工作线程；不保证越大越快。
--no-video跳过编码仅输出结构化结果。--max-frames N可快速测试。
--amp可选混合精度；默认关闭以保持边界检测稳定，使用前应验证候选和误检变化。
CUDA不可用会明确失败，不会静默切回CPU。Windows CPU解码/编码可能仍是瓶颈；未使用NVENC、TensorRT或训练加速。

## 输出

outputs/<视频名>/：
- analysis.mp4：原图分辨率视频，球轨迹、右上角落点 minimap 和最近一次最优球速估计，无音频；不画球拍、pose 或旋转。
- predictions.csv：融合后的原图像素坐标。
- full_predictions.csv / far_predictions.csv：两个窗口的原始检测。
- fusion_audit.csv：逐帧来源和选择原因。
- candidates.json / bounce_scores.csv：原始CatBoost候选和逐帧评分。
- rackets.json：实际推理的球拍关键点、框和模型哈希。
- events.json：复核证据和事件分组。
- run.json：耗时、配置、阶段时间、坐标约定。

所有路径可用 --output-dir 指定。已有结果默认拒绝覆盖；需要时显式 --overwrite。

## 独立目录结构

- rally_analysis.cmd / run.py / cli.py：同一个入口的Windows启动器与分发逻辑。
- runtime/：官方CPython3.11.9嵌入式运行体和本地依赖；manifest.json记录版本。
- RacketVision/：只保留实际需要的BallTrack模型源码、RacketPose推理配置、3个权重和上游LICENSE。
- models/bounce/：CatBoost权重及来源记录。
- calibrations/：机位标定。
- engine.py / pipeline.py / video_io.py：常驻CUDA模型、有界异步任务和视频IO。
- compat/：集中保存旧算法实现/历史工具，根目录旧脚本只作兼容转发；日常使用无需调用它们。
- tests/：可随目录运行的测试。
- tools/bundle_runtime.py：打包构建工具，运行分析不需要它或外部构建环境。

不包含训练数据集、不需要原始工作区的虚拟环境或完整Git仓库。历史kjfk比较和数据下载脚本
仅为归档保留，不是此运行体的支持入口；它们各自的NumPy2/下载依赖不混入稳定分析环境。
NVIDIA驱动、Windows系统DLL、输入视频仍由电脑提供。不要删除runtime或权重子目录。

## 本机验证

RTX4080、FP32、双窗口[650,250,650,320]，完整300帧point_0003：任务阶段约23.3秒，
含模型加载和一次最终编码（不含Python进程启动）。这不是跨硬件性能保证，也不是准确率。
最终分组与原增强流程相同：79、210、262有支持，141、166待确认。
验证输出outputs/validation/point_0003/。
两个90帧任务从工作区外目录启动成功，模型各加载一次，视频编码与后续GPU工作发生重叠。
测试详情以self-test和各批次summary.json为准；不依赖外部reference checkout的特征对照测试会跳过。

运行体当前约6 GiB，主要是PyTorch/CUDA库与权重；不需要额外激活环境。bundle_manifest.json记录4个模型权重的相对路径、大小和SHA256，runtime/manifest.json记录依赖版本。依赖自身的许可证文件保留在各包/元数据目录中，上游模型源码LICENSE位于RacketVision/LICENSE。

## Calibrate then analyze with one PowerShell command

    .\test_analysis.ps1 -VideoPath "C:\videos\tennis.mp4" -CameraId court1

An existing calibration for CameraId is reused automatically. If none exists, manual four-corner confirmation is required before analysis starts. Cancelling preserves the previous calibration and does not start analysis. Use -Overwrite to select four new corners and replace calibration/results, -OverwriteResults to replace analysis outputs while reusing calibration, -FarRoi 650,250,650,320 for an explicit far window, and -WhatIf to preview without changing files or opening the UI. The root-level test_analysis.ps1 also locates this workflow if the folder is named rally_analysis_ai.

## 目标球场过滤与双窗口融合

默认继续使用整场窗口（覆盖近端）+ 远端细节窗口；两者独立检测后融合为一条轨迹，再进行 CatBoost 和球拍证据复核，并非近远端各自独立判 bounce。

标定现在用于逐窗口热图候选过滤：球场横向默认保留 1 米余量、底线各保留 2 米，并向图像上方扩展 25% 图像高度以容纳空中球。随后按已建立轨迹的速度与连续性选择窗口；窗口冲突不再固定优先整场。无可信检测时输出缺失，不跨邻场连接轨迹。原始模型输入图像不做黑色遮罩。

--court-margin-m 1 可调整横向余量（0..5 米），--ball-lift .25 可调整向上飞行余量（0..0.75 图像高度）。高吊球需同时确认检测窗口覆盖范围。这是固定机位的空间先验，画面投影重叠时仍不能保证区分所有邻场球。

court_region.png 显示允许区域（白色），fusion_audit.csv 记录窗口来源及空间/运动拒绝原因；full_predictions.csv / far_predictions.csv 是空间过滤后的各窗口结果。

2026-10-29 CUDA 完整回放：point_0021 共199帧，第165帧由邻场 (238,438) 改为本场 (829,431)，旧候选131/157消失，仅96待确认；point_0003 共300帧，79/210/262仍有支持、141/166待确认。输出位于 outputs/court_guard_validation 与 outputs/court_guard_regression。self-test 共36项，35通过，1项参考对照跳过。

## 球速估计与落点 minimap

正常 analyze 流程默认在复核之后计算球速，无需额外参数。analysis.mp4 右上角参考转播样式显示半透明球场图，最多保留最近8个有支持的落点，白圈突出最新落点；绿/橙按远近半场区分。uncertain 候选也作为最可能落点显示并记录原始证据等级；likely_racket_hit 仍不画成落点。

落点在事件帧后3帧出现，速度在事件帧后6帧更新，以反映现有 bounce/球拍证据流程的后续观测。最新一次有效速度持续保留到下一次有效测速；尚无有效速度显示 -- km/h。证据不足时也选择最佳可用数值：先尝试较弱支持的弹道拟合，再使用场地平面位移近似；完全无法测量时以上次速度作为先验，并记录 previous_speed_prior。没有任何当前或历史可用观测时才显示 --。面板标签仅为 Speed，使用原生分辨率 TrueType 抗锯齿字体，单位 km/h。这是离线分析的观测落点估计，不是飞行途中提前预报未来落点。

新增 speeds.json：每个非击球候选保存 status/reason、落点、landing_available_frame、available_frame、所有有效 candidates 和 best_prediction。按重投影误差、落点一致性、内点比例和时间跨度评分，选 selection_cost 最小的一项；每次显示该项速度，不按速度最大值选择。run.json 保存有效 speed_estimates 数量。

算法参考 Zhang & Li (2025), [Tennis ball detection based on YOLOv5 with tensorrt](https://doi.org/10.1038/s41598-025-06365-3) 的场地标定、三维坐标和 RANSAC 思路。论文使用多相机；本实现另外构建单目落点约束重力弹道模型，不能宣称复现论文的多相机精度。空中球不直接通过地面 homography 计算速度。

模型假设固定相机、标准尺寸球场、零畸变/方形像素、主点位于画面中心，并由场地 homography 估计焦距。可在标定 JSON 添加 camera_matrix（3x3 像素单位内参）替代焦距估计；输入应已去畸变。相机或轨迹退化、长漏检、不确定 bounce 会触发低置信度最佳可用估计，保存 estimated_low_confidence、confidence、reason 和候选 method。非物理解不会作为有效弹道；可回退到明确标记的场地平面位移近似，这种近似并非空中球三维速度。最多跨2帧缺失但只拟合真实观测，不插值。

速度是最优落地前轨迹段起点的三维速度，单位 km/h（JSON 同存 mph），没有可靠击球时间时不等于出拍瞬间速度。模型未考虑空气阻力和旋转；置信度和估计方法保存在 JSON 中，画面只显示 Speed 与数值。现有球拍证据复核继续在内部辅助区分 bounce/hit；本次未增加人体 pose、旋转模型或相关 overlay。

验证：self-test 包含已知三维速度、30/60/120 FPS、异常点、短漏检、无效相机和显示时序测试。outputs/speed_validation/ 下的三个历史片段保存重渲染 analysis.mp4、speeds.json、preview.jpg；这些是真实视频功能验证，尚无雷达测速真值，不能据此推断准确率。

本次完整验证：self-test 共50项，49通过、1项外部参考对照跳过；RTX4080 完整300帧流程成功，run.json 记录2次有效测速，输出视频逐帧解码确认300帧。完整输出：outputs/speed_end_to_end/point_0003/。

2026-11-05 显示修订：约4.70秒（第141帧）的原始候选不再因球拍证据不足被排除，minimap记录并显示该落点，速度最佳拟合约34.4 km/h（低置信度）。第166/262帧同样显示最佳可用速度。新视频及该时刻截图：outputs/speed_revision/point_0003/analysis.mp4、bounce_4s.jpg。52项测试中51通过、1项参考对照跳过；三个重渲染片段逐帧解码确认300/177/199帧。

## 短漏检击球接触复核与转播卡片字体

候选附近只有1–2帧漏检且方向变化至少90度时，会分别使用前后3帧观测构建有加速度约束的二次曲线，提出短缺失段内的接触区域。不会把这类外推坐标写进 predictions.csv。对这些区域最多做3个 RTMPose 局部提议；它们明确标为 trajectory_guided_pose、bbox_score=0，不能冒充 detector 的真实检测。

仅在拍头平均置信度至少0.5、两个端点至少0.45，且邻近3帧内存在独立检测到的同区域球拍时保留提议。最终还需两侧轨迹在同一拍头附近汇合、间距小于一拍头长度，并与至少60度的运动反转共同满足，才标为 likely_racket_hit。nearest_gap_racket_contact 保存接触帧、两侧假设位置、支持帧和 proposal_verified_by_frame；原始候选帧与模型分数保留。

point_0003 的166候选：167–168漏检，恢复到168的接触姿态，并由170的实际球拍检测佐证。候选现在判为 likely_racket_hit，不进入 minimap 或落点测速；不是把整段低证据事件一概过滤，141仍保留。

卡片按参考图使用 Segoe UI Bold：左侧绿色 Speed，右侧白色粗体数值及下方独立 km/h，中间竖线，圆角半透明背景。minimap 与速度卡片分离；不增加球拍或人体姿态画面。
