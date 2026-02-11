import AppKit
import Foundation
import ScreenCaptureKit
import AVFoundation
import Accelerate

// MARK: - CLI Argument Parsing

struct Args {
    var appName: String?
    var sampleRate: Int = 16000
    var channels: Int = 1
    var listApps: Bool = false
}

func parseArgs() -> Args {
    var args = Args()
    let argv = CommandLine.arguments
    var i = 1
    while i < argv.count {
        guard i < argv.count else { break }
        switch argv[i] {
        case "--app":
            i += 1
            if i < argv.count { args.appName = argv[i] }
        case "--rate":
            i += 1
            if i < argv.count { args.sampleRate = Int(argv[i]) ?? 16000 }
        case "--channels":
            i += 1
            if i < argv.count { args.channels = Int(argv[i]) ?? 1 }
        case "--list-apps":
            args.listApps = true
        default:
            break
        }
        i += 1
    }
    return args
}

// MARK: - Low-Pass FIR Filter

/// Generate a windowed-sinc low-pass filter kernel.
/// cutoffHz: cutoff frequency, sampleRate: source sample rate, taps: must be odd
func makeLowPassKernel(cutoffHz: Double, sampleRate: Double, taps: Int) -> [Float] {
    let n = taps
    let mid = n / 2
    let fc = cutoffHz / sampleRate
    var kernel = [Float](repeating: 0, count: n)
    for i in 0..<n {
        let x = Double(i - mid)
        if i == mid {
            kernel[i] = Float(2.0 * fc)
        } else {
            // sinc
            let sinc = sin(2.0 * .pi * fc * x) / (.pi * x)
            // Hann window
            let window = 0.5 * (1.0 - cos(2.0 * .pi * Double(i) / Double(n - 1)))
            kernel[i] = Float(sinc * window)
        }
    }
    // Normalize
    let sum = kernel.reduce(0, +)
    if sum > 0 {
        for i in 0..<n { kernel[i] /= sum }
    }
    return kernel
}

// MARK: - Audio Capture Delegate

class AudioOutputHandler: NSObject, SCStreamOutput {
    let sampleRate: Int
    let channels: Int
    let outputQueue = DispatchQueue(label: "langlistn.audio.output")

    // Pre-computed anti-alias filter for 48kHz → 16kHz (cutoff at 7.5kHz)
    private var antiAliasKernel: [Float] = []
    private let filterTaps = 33  // odd, ~33 taps is good for 3:1 decimation

    init(sampleRate: Int, channels: Int) {
        self.sampleRate = sampleRate
        self.channels = channels
        super.init()
        // Pre-compute filter kernel (assume 48kHz source)
        self.antiAliasKernel = makeLowPassKernel(
            cutoffHz: Double(sampleRate) / 2.0 * 0.95,  // slight margin below Nyquist
            sampleRate: 48000.0,
            taps: filterTaps
        )
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio else { return }
        guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else {
            return
        }

        var length = 0
        var dataPointer: UnsafeMutablePointer<Int8>?
        CMBlockBufferGetDataPointer(blockBuffer, atOffset: 0, lengthAtOffsetOut: nil, totalLengthOut: &length, dataPointerOut: &dataPointer)

        guard let dataPointer = dataPointer, length > 0 else { return }

        guard let formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(formatDesc) else { return }

        let srcSampleRate = asbd.pointee.mSampleRate
        let srcChannels = Int(asbd.pointee.mChannelsPerFrame)
        let bytesPerFrame = Int(asbd.pointee.mBytesPerFrame)
        let isFloat = asbd.pointee.mFormatFlags & kAudioFormatFlagIsFloat != 0
        let frameCount = length / max(bytesPerFrame, 1)

        // Step 1: Convert to float32 mono
        var floatSamples = [Float]()
        floatSamples.reserveCapacity(frameCount)

        if isFloat && bytesPerFrame == srcChannels * 4 {
            // Interleaved float32 (most common from ScreenCaptureKit)
            let floatPtr = UnsafeRawPointer(dataPointer).bindMemory(to: Float.self, capacity: frameCount * srcChannels)
            for f in 0..<frameCount {
                var sample: Float = 0
                for c in 0..<srcChannels {
                    sample += floatPtr[f * srcChannels + c]
                }
                sample /= Float(srcChannels)
                floatSamples.append(max(-1.0, min(1.0, sample)))
            }
        } else if isFloat {
            // Non-interleaved float32 — use AudioBufferList for correctness
            // Fallback: treat as interleaved with bytesPerFrame=4 (one channel)
            let floatPtr = UnsafeRawPointer(dataPointer).bindMemory(to: Float.self, capacity: length / 4)
            let count = length / 4
            for i in 0..<count {
                floatSamples.append(max(-1.0, min(1.0, floatPtr[i])))
            }
        } else {
            // int16 — convert to float
            let int16Ptr = UnsafeRawPointer(dataPointer).bindMemory(to: Int16.self, capacity: length / 2)
            let count = length / 2
            for i in 0..<count {
                floatSamples.append(Float(int16Ptr[i]) / 32768.0)
            }
        }

        guard !floatSamples.isEmpty else { return }

        // Step 2: Anti-alias filter + decimate if needed
        var outputSamples: [Float]
        if Int(srcSampleRate) != sampleRate && !floatSamples.isEmpty {
            let ratio = srcSampleRate / Double(sampleRate)
            let outFrames = Int(Double(floatSamples.count) / ratio)
            guard outFrames > 0 else { return }

            // Apply anti-alias FIR filter using vDSP
            let padded = [Float](repeating: 0, count: filterTaps / 2) + floatSamples + [Float](repeating: 0, count: filterTaps / 2)
            var filtered = [Float](repeating: 0, count: floatSamples.count)
            vDSP_conv(padded, 1, antiAliasKernel, 1, &filtered, 1, vDSP_Length(floatSamples.count), vDSP_Length(filterTaps))

            // Decimate (pick every Nth sample with linear interpolation for fractional ratio)
            outputSamples = [Float]()
            outputSamples.reserveCapacity(outFrames)
            for i in 0..<outFrames {
                let srcIdx = Double(i) * ratio
                let idx = Int(srcIdx)
                let frac = Float(srcIdx - Double(idx))
                if idx + 1 < filtered.count {
                    outputSamples.append(filtered[idx] + frac * (filtered[idx + 1] - filtered[idx]))
                } else if idx < filtered.count {
                    outputSamples.append(filtered[idx])
                }
            }
        } else {
            outputSamples = floatSamples
        }

        // Step 3: Convert to int16 PCM and write to stdout
        var pcmData = Data()
        pcmData.reserveCapacity(outputSamples.count * 2)
        for sample in outputSamples {
            let clamped = max(-1.0, min(1.0, sample))
            var int16Val = Int16(clamped * 32767.0)
            pcmData.append(Data(bytes: &int16Val, count: 2))
        }

        // Write is already serialized — callback runs on outputQueue
        pcmData.withUnsafeBytes { ptr in
            let _ = fwrite(ptr.baseAddress, 1, ptr.count, stdout)
            fflush(stdout)
        }
    }
}

// MARK: - Stream Delegate

class StreamDelegate: NSObject, SCStreamDelegate {
    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("Error: stream stopped — \(error.localizedDescription)\n", stderr)
        exit(2)
    }
}

// MARK: - App Delegate

class AppDelegate: NSObject, NSApplicationDelegate {
    let args: Args
    var stream: SCStream?
    var handler: AudioOutputHandler?
    var streamDelegate: StreamDelegate?

    init(args: Args) {
        self.args = args
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Hide from Dock
        NSApp.setActivationPolicy(.accessory)

        Task {
            await self.run()
        }
    }

    func run() async {
        do {
            let content = try await SCShareableContent.current

            if args.listApps {
                let apps = content.applications
                    .filter { $0.applicationName.count > 0 }
                    .map { $0.applicationName }
                let unique = Array(Set(apps)).sorted()
                for name in unique {
                    print(name)
                }
                exit(0)
            }

            guard let appName = args.appName else {
                fputs("Error: --app required\n", stderr)
                exit(1)
            }

            // Prefer exact match, fall back to substring match
            let app = content.applications.first(where: {
                $0.applicationName.caseInsensitiveCompare(appName) == .orderedSame
            }) ?? content.applications.first(where: {
                $0.applicationName.localizedCaseInsensitiveContains(appName)
            })
            guard let app = app else {
                fputs("Error: App '\(appName)' not found or not running. Use --list-apps to see available apps.\n", stderr)
                exit(1)
            }

            let appFilter = SCContentFilter(
                display: content.displays[0],
                including: [app],
                exceptingWindows: []
            )

            let config = SCStreamConfiguration()
            config.capturesAudio = true
            config.excludesCurrentProcessAudio = true
            config.sampleRate = 48000
            config.channelCount = 2
            // Minimize video overhead — audio-only workaround requires .screen output
            config.width = 2
            config.height = 2
            config.minimumFrameInterval = CMTime(value: 1, timescale: 1)  // 1 FPS

            let streamDel = StreamDelegate()
            self.streamDelegate = streamDel
            let s = SCStream(filter: appFilter, configuration: config, delegate: streamDel)
            let h = AudioOutputHandler(sampleRate: args.sampleRate, channels: args.channels)
            self.handler = h
            self.stream = s

            // Audio callbacks require a .screen output to be registered (ScreenCaptureKit limitation)
            try s.addStreamOutput(h, type: .screen, sampleHandlerQueue: .global())
            try s.addStreamOutput(h, type: .audio, sampleHandlerQueue: h.outputQueue)

            try await s.startCapture()
            fputs("Capturing audio from: \(app.applicationName)\n", stderr)

        } catch {
            let msg = error.localizedDescription
            if msg.contains("permission") || msg.contains("denied") || msg.contains("TCCDeny") {
                fputs("Error: Screen Recording permission denied. Grant in System Settings > Privacy.\n", stderr)
                exit(3)
            }
            fputs("Error: \(msg)\n", stderr)
            exit(1)
        }
    }
}

// MARK: - Main

let args = parseArgs()

signal(SIGTERM) { _ in
    fflush(stdout)
    exit(0)
}
signal(SIGINT) { _ in
    fflush(stdout)
    exit(0)
}

let delegate = AppDelegate(args: args)
let app = NSApplication.shared
app.delegate = delegate
app.run()
