// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/crash_handler.h"

#ifdef _WIN32

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <string>
// clang-format off
#include <windows.h>
#include <dbghelp.h>
// clang-format on

#include "common/logging/backend.h"
#include "common/path_util.h"

#pragma comment(lib, "dbghelp.lib")

namespace Common {

namespace {

// Only the first thread to fault gets to write a report; everything else bails out so we never
// recurse into the handler (e.g. if symbol resolution itself faults) or interleave two dumps.
std::atomic<bool> g_in_handler{false};

std::filesystem::path CrashReportDir() {
    std::filesystem::path dir;
    try {
        dir = FS::GetUserPath(FS::PathType::LogDir);
    } catch (...) {
        dir.clear();
    }
    if (dir.empty()) {
        std::error_code ec;
        dir = std::filesystem::temp_directory_path(ec);
    }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir;
}

const char* BaseName(const char* path) {
    const char* base = path;
    for (const char* p = path; *p; ++p) {
        if (*p == '\\' || *p == '/') {
            base = p + 1;
        }
    }
    return base;
}

void WriteMinidump(const std::filesystem::path& path, EXCEPTION_POINTERS* info) {
    HANDLE file = CreateFileW(path.wstring().c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return;
    }
    MINIDUMP_EXCEPTION_INFORMATION mei{};
    mei.ThreadId = GetCurrentThreadId();
    mei.ExceptionPointers = info;
    mei.ClientPointers = FALSE;
    const auto type = static_cast<MINIDUMP_TYPE>(MiniDumpWithThreadInfo | MiniDumpWithDataSegs |
                                                 MiniDumpWithIndirectlyReferencedMemory);
    MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), file, type,
                      info ? &mei : nullptr, nullptr, nullptr);
    CloseHandle(file);
}

void WriteBacktrace(std::FILE* out, EXCEPTION_POINTERS* info) {
    const HANDLE process = GetCurrentProcess();
    const HANDLE thread = GetCurrentThread();

    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME);
    SymInitialize(process, nullptr, TRUE);

    CONTEXT ctx = *info->ContextRecord; // StackWalk64 mutates the context, so work on a copy.
    STACKFRAME64 frame{};
    DWORD machine;
#if defined(_M_X64) || defined(__x86_64__)
    machine = IMAGE_FILE_MACHINE_AMD64;
    frame.AddrPC.Offset = ctx.Rip;
    frame.AddrFrame.Offset = ctx.Rbp;
    frame.AddrStack.Offset = ctx.Rsp;
#elif defined(_M_ARM64) || defined(__aarch64__)
    machine = IMAGE_FILE_MACHINE_ARM64;
    frame.AddrPC.Offset = ctx.Pc;
    frame.AddrFrame.Offset = ctx.Fp;
    frame.AddrStack.Offset = ctx.Sp;
#else
#error "Unsupported architecture for crash backtrace"
#endif
    frame.AddrPC.Mode = AddrModeFlat;
    frame.AddrFrame.Mode = AddrModeFlat;
    frame.AddrStack.Mode = AddrModeFlat;

    alignas(SYMBOL_INFO) char sym_storage[sizeof(SYMBOL_INFO) + 512];
    auto* symbol = reinterpret_cast<SYMBOL_INFO*>(sym_storage);

    for (int i = 0; i < 64; ++i) {
        if (!StackWalk64(machine, process, thread, &frame, &ctx, nullptr,
                         SymFunctionTableAccess64, SymGetModuleBase64, nullptr)) {
            break;
        }
        const DWORD64 addr = frame.AddrPC.Offset;
        if (addr == 0) {
            break;
        }

        char module_name[MAX_PATH] = "<unknown>";
        if (const DWORD64 module_base = SymGetModuleBase64(process, addr)) {
            GetModuleFileNameA(reinterpret_cast<HMODULE>(module_base), module_name, MAX_PATH);
        }

        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        symbol->MaxNameLen = 512;
        DWORD64 sym_disp = 0;
        const char* sym_name = "<no symbol>";
        if (SymFromAddr(process, addr, &sym_disp, symbol)) {
            sym_name = symbol->Name;
        }

        IMAGEHLP_LINE64 line{};
        line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
        DWORD line_disp = 0;
        if (SymGetLineFromAddr64(process, addr, &line_disp, &line)) {
            std::fprintf(out, "#%-2d 0x%016llx  %s!%s+0x%llx  (%s:%lu)\n", i,
                         static_cast<unsigned long long>(addr), BaseName(module_name), sym_name,
                         static_cast<unsigned long long>(sym_disp), line.FileName,
                         line.LineNumber);
        } else {
            std::fprintf(out, "#%-2d 0x%016llx  %s!%s+0x%llx\n", i,
                         static_cast<unsigned long long>(addr), BaseName(module_name), sym_name,
                         static_cast<unsigned long long>(sym_disp));
        }
    }

    SymCleanup(process);
}

const char* ExceptionName(DWORD code) {
    switch (code) {
    case EXCEPTION_ACCESS_VIOLATION:
        return "EXCEPTION_ACCESS_VIOLATION";
    case EXCEPTION_ILLEGAL_INSTRUCTION:
        return "EXCEPTION_ILLEGAL_INSTRUCTION";
    case EXCEPTION_STACK_OVERFLOW:
        return "EXCEPTION_STACK_OVERFLOW";
    case EXCEPTION_INT_DIVIDE_BY_ZERO:
        return "EXCEPTION_INT_DIVIDE_BY_ZERO";
    case EXCEPTION_PRIV_INSTRUCTION:
        return "EXCEPTION_PRIV_INSTRUCTION";
    case EXCEPTION_IN_PAGE_ERROR:
        return "EXCEPTION_IN_PAGE_ERROR";
    case EXCEPTION_DATATYPE_MISALIGNMENT:
        return "EXCEPTION_DATATYPE_MISALIGNMENT";
    case 0xC0000409:
        return "STATUS_STACK_BUFFER_OVERRUN";
    case 0xE06D7363:
        return "C++ exception (unhandled)";
    default:
        return "Unknown";
    }
}

LONG WINAPI Backstop(EXCEPTION_POINTERS* info) {
    WriteCrashReport(info, "backstop");
    return EXCEPTION_EXECUTE_HANDLER; // terminate the process
}

} // namespace

bool WriteCrashReport(EXCEPTION_POINTERS* info, const char* origin) {
    bool expected = false;
    if (!g_in_handler.compare_exchange_strong(expected, true)) {
        return false;
    }

    const DWORD code = (info && info->ExceptionRecord) ? info->ExceptionRecord->ExceptionCode : 0;
    const void* addr =
        (info && info->ExceptionRecord) ? info->ExceptionRecord->ExceptionAddress : nullptr;

    const auto dir = CrashReportDir();
    char stamp[96];
    std::snprintf(stamp, sizeof(stamp), "crash_%lu_%llu", GetCurrentProcessId(),
                  static_cast<unsigned long long>(GetTickCount64()));

    const auto dmp_path = dir / (std::string(stamp) + ".dmp");
    const auto txt_path = dir / (std::string(stamp) + ".txt");

    if (info) {
        WriteMinidump(dmp_path, info);
    }

    if (std::FILE* out = _wfopen(txt_path.wstring().c_str(), L"w")) {
        std::fprintf(out, "shadPS4 crash report (origin: %s)\n", origin ? origin : "?");
        std::fprintf(out, "Exception: %s (0x%08lx) at 0x%p\n", ExceptionName(code), code, addr);
        if (code == EXCEPTION_ACCESS_VIOLATION && info && info->ExceptionRecord &&
            info->ExceptionRecord->NumberParameters >= 2) {
            const ULONG_PTR rw = info->ExceptionRecord->ExceptionInformation[0];
            const ULONG_PTR fault = info->ExceptionRecord->ExceptionInformation[1];
            std::fprintf(out, "Access violation: %s address 0x%llx\n",
                         rw == 1 ? "write to" : (rw == 8 ? "execute at" : "read from"),
                         static_cast<unsigned long long>(fault));
        }
        std::fprintf(out, "Minidump: %s\n\n", dmp_path.string().c_str());
        std::fprintf(out, "Faulting thread backtrace:\n");
        if (info) {
            WriteBacktrace(out, info);
        }
        std::fflush(out);
        std::fclose(out);
    }

    std::fprintf(stderr,
                 "\n==== shadPS4 crashed: %s (0x%08lx) at 0x%p ====\nCrash report: %s\nMinidump:     "
                 "%s\n",
                 ExceptionName(code), code, addr, txt_path.string().c_str(),
                 dmp_path.string().c_str());
    std::fflush(stderr);

    // Drain and flush all log backends so the lead-up to the crash is on disk. The async logger
    // runs on its own thread, so this is safe to call from the faulting thread.
    try {
        Log::Stop();
    } catch (...) {
    }

    return true;
}

void InstallCrashHandler() {
    // Reserve a slice of stack so the handler can still run (and write a dump) after a stack
    // overflow, which otherwise leaves no room to do any work.
    ULONG guarantee = 64 * 1024;
    SetThreadStackGuarantee(&guarantee);
    SetUnhandledExceptionFilter(Backstop);
}

} // namespace Common

#else

namespace Common {
void InstallCrashHandler() {
    // TODO: POSIX backtrace/minidump support. The native crash path on Linux/macOS currently goes
    // through Core's signal handler and UNREACHABLE_MSG, which already aborts with a message.
}
} // namespace Common

#endif
