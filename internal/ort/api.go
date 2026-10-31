package ort

// Indices into the OrtApi function table. OrtApi is append-only and
// GetApi(APIVersion) returns the table of that version, so the offsets of
// everything below are fixed for every runtime that implements API 22 or
// newer. Source: include/onnxruntime/core/session/onnxruntime_c_api.h at
// onnxruntime v1.22.0 (struct OrtApi, entries counted in declaration order).
// Every entry is exercised against a live library by ort_test.go.
const (
	idxGetErrorCode                   = 1
	idxGetErrorMessage                = 2
	idxCreateEnv                      = 3
	idxCreateSession                  = 7
	idxRun                            = 9
	idxCreateSessionOptions           = 10
	idxDisableCpuMemArena             = 19
	idxSetSessionGraphOptLevel        = 23
	idxSetIntraOpNumThreads           = 24
	idxSetInterOpNumThreads           = 25
	idxSessionGetInputCount           = 30
	idxSessionGetOutputCount          = 31
	idxSessionGetInputName            = 36
	idxSessionGetOutputName           = 37
	idxCreateRunOptions               = 39
	idxCreateTensorAsOrtValue         = 48
	idxGetTensorMutableData           = 51
	idxGetTensorElementType           = 60
	idxGetDimensionsCount             = 61
	idxGetDimensions                  = 62
	idxGetTensorShapeElementCount     = 64
	idxGetTensorTypeAndShape          = 65
	idxGetValueType                   = 67
	idxAllocatorFree                  = 76
	idxGetAllocatorWithDefaultOptions = 78
	idxReleaseEnv                     = 92
	idxReleaseStatus                  = 93
	idxReleaseSession                 = 95
	idxReleaseValue                   = 96
	idxReleaseRunOptions              = 97
	idxReleaseTensorTypeAndShape      = 99
	idxReleaseSessionOptions          = 100
	idxGetAvailableProviders          = 125
	idxReleaseAvailableProviders      = 126
	idxAddSessionConfigEntry          = 130
	idxAddRunConfigEntry              = 165
	idxAppendCUDAProviderV2           = 204 // SessionOptionsAppendExecutionProvider_CUDA_V2
	idxCreateCUDAProviderOptions      = 205
	idxUpdateCUDAProviderOptions      = 206
	idxReleaseCUDAProviderOptions     = 208
)

// ProviderCUDA is the name GetAvailableProviders reports for a runtime built
// with the CUDA execution provider.
const ProviderCUDA = "CUDAExecutionProvider"

// APIVersion is the ORT C API version this package is written against.
const APIVersion = 22

// ONNXTensorElementDataType values used here.
const (
	elemFloat32 = 1
	elemInt64   = 7
	elemBool    = 9
)

// ONNXType of an OrtValue that holds a tensor.
const onnxTypeTensor = 1

// Graph optimization level ORT_ENABLE_ALL.
const optAll = 99

const logError = 3 // OrtLoggingLevel ORT_LOGGING_LEVEL_ERROR
