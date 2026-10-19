# SPDX-License-Identifier: Apache-2.0
"""Offline export audit for the pinned ELF64/PE32+ x64 package inputs.

This reads file-backed tables without loading foreign native code. It is not a
general executable validator or proof of publisher/source authenticity.
Formats: https://gabi.xinuos.com/elf/05-symtab.html and
https://learn.microsoft.com/en-us/windows/win32/debug/pe-format
"""
import struct


def exported_names(path, rid):
    data = path.read_bytes()

    def span(offset, size):
        if offset < 0 or size < 0 or offset > len(data) or size > len(data) - offset:
            raise ValueError("native export table outside file")
        return offset

    def unpack(fmt, offset):
        return struct.unpack_from(fmt, data, span(offset, struct.calcsize(fmt)))

    def string(offset, end):
        span(offset, end - offset)
        terminator = data.find(b"\0", offset, end)
        if terminator < 0:
            raise ValueError("unterminated native export name")
        try:
            return data[offset:terminator].decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("non-ASCII native export name") from error

    names = []
    if rid == "linux-x64":
        if data[:7] != b"\x7fELF\x02\x01\x01" or unpack("<HH", 16) != (3, 62):
            raise ValueError("native exports require an ELF64 x64 shared library")
        section_offset = unpack("<Q", 40)[0]
        section_size, section_count = unpack("<HH", 58)
        if section_size != 64 or not 0 < section_count < 0xff00:
            raise ValueError("unsupported ELF export section table")
        span(section_offset, section_size * section_count)
        sections = [unpack("<IIQQQQIIQQ", section_offset + i * section_size)
                    for i in range(section_count)]
        tables = [s for s in sections if s[1] == 11]  # SHT_DYNSYM, not .symtab.
        if len(tables) != 1:
            raise ValueError("expected one ELF dynamic export table")
        symbols = tables[0]
        if symbols[9] != 24 or symbols[5] % 24 or symbols[6] >= section_count:
            raise ValueError("invalid ELF export table entries/link")
        strings = sections[symbols[6]]
        if strings[1] != 3:  # SHT_STRTAB
            raise ValueError("invalid ELF export string table")
        span(symbols[4], symbols[5]); span(strings[4], strings[5])
        for offset in range(symbols[4], symbols[4] + symbols[5], 24):
            name, info, visibility, section, value, size = unpack("<IBBHQQ", offset)
            if name >= strings[5]:
                raise ValueError("ELF export name outside string table")
            # Defined global/weak symbols with default/protected visibility.
            if section and info >> 4 in (1, 2) and visibility & 7 in (0, 3):
                text = string(strings[4] + name, strings[4] + strings[5])
                if text.startswith("fy_"):
                    if info & 15 != 2 or not 0 < section < section_count:
                        raise ValueError("FactsYOLO ELF export is not a defined function")
                    names.append(text)
    elif rid == "win-x64":
        pe = unpack("<I", 60)[0]
        if data[:2] != b"MZ" or data[pe:pe + 4] != b"PE\0\0":
            raise ValueError("native exports require a PE DLL")
        machine, section_count = unpack("<HH", pe + 4)
        optional_size, flags = unpack("<HH", pe + 20)
        optional = pe + 24
        span(optional, optional_size)
        if machine != 0x8664 or not flags & 0x2000 or optional_size < 120 or unpack("<H", optional)[0] != 0x20b:
            raise ValueError("native exports require a PE32+ x64 DLL")
        if not section_count or unpack("<I", optional + 108)[0] < 1:
            raise ValueError("missing PE export directory")
        directory, directory_size = unpack("<II", optional + 112)
        if not directory or directory_size < 40:
            raise ValueError("missing PE export directory")
        start = optional + optional_size
        span(start, section_count * 40)
        sections = [unpack("<IIII", start + i * 40 + 8) for i in range(section_count)]

        def rva(address, size):
            matches = []
            for virtual_size, virtual, raw_size, raw in sections:
                delta = address - virtual
                if 0 <= delta < raw_size and size <= raw_size - delta:
                    span(raw, raw_size)
                    matches.append((raw + delta, raw + raw_size))
            if len(matches) != 1:
                raise ValueError("PE export RVA is not uniquely file-backed")
            return matches[0]

        export, _ = rva(directory, directory_size)
        functions, count, addresses, pointers, ordinals = unpack("<IIIII", export + 20)
        if not functions or not count or count > functions:
            raise ValueError("invalid PE export counts")
        address_table, _ = rva(addresses, functions * 4)
        name_table, _ = rva(pointers, count * 4)
        ordinal_table, _ = rva(ordinals, count * 2)
        for i in range(count):
            pointer, end = rva(unpack("<I", name_table + 4 * i)[0], 1)
            text = string(pointer, end)
            ordinal = unpack("<H", ordinal_table + 2 * i)[0]
            if ordinal >= functions:
                raise ValueError("PE export ordinal outside address table")
            address = unpack("<I", address_table + 4 * ordinal)[0]
            if text.startswith("fy_"):
                if not address or directory <= address < directory + directory_size:
                    raise ValueError("FactsYOLO PE export is absent or forwarded")
                rva(address, 1)
                names.append(text)
    else:
        raise ValueError("unsupported native export RID")
    if len(names) != len(set(names)):
        raise ValueError("duplicate FactsYOLO native exports")
    return sorted(names)
