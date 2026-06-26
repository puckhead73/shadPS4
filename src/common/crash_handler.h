// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifdef _WIN32
struct _EXCEPTION_POINTERS;
#endif

namespace Common {

/// Installs a process-wide last-resort crash handler. On an otherwise unhandled fatal
/// exception this writes a minidump (.dmp) plus a symbolized backtrace (.txt) to the log
/// directory and flushes the logger, so native crashes leave a postmortem trail instead of
/// a silently truncated log. Safe to call once, early in main().
void InstallCrashHandler();

#ifdef _WIN32
/// Writes a crash report (minidump + symbolized backtrace) for the given exception and flushes
/// logs. Reentrancy-guarded: only the first caller per process produces a report. `origin` is a
/// short tag identifying where the report was triggered from (e.g. "veh", "backstop").
/// Returns true if this call produced the report.
bool WriteCrashReport(_EXCEPTION_POINTERS* info, const char* origin);
#endif

} // namespace Common
