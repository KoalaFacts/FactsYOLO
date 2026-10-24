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
        versions = [s for s in sections if s[1] == 0x6fffffff]  # SHT_GNU_versym
        if len(versions) > 1:
            raise ValueError("multiple ELF symbol version tables")
        if versions:
            version = versions[0]
            if (version[6] != sections.index(symbols) or version[9] != 2
                    or version[5] != symbols[5] // 24 * 2):
                raise ValueError("invalid ELF symbol version table")
            span(version[4], version[5])
        # Section headers are only an inventory. Lookup uses PT_DYNAMIC and
        # virtual addresses mapped through PT_LOAD, so require agreement.
        program_offset = unpack("<Q", 32)[0]
        program_size, program_count = unpack("<HH", 54)
        if program_size != 56 or not 0 < program_count < 0xffff:
            raise ValueError("invalid ELF export program table")
        span(program_offset, program_size * program_count)
        programs = [unpack("<IIQQQQQQ", program_offset + i * 56) for i in range(program_count)]
        loads = [p for p in programs if p[0] == 1]
        for p in loads:
            if p[5] > p[6] or p[3] + p[6] > 1 << 64:
                raise ValueError("invalid ELF export load segment")
            span(p[2], p[5])
        ranges = sorted((p[3], p[3] + p[6]) for p in loads if p[6])
        if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
            raise ValueError("overlapping ELF export load mappings")

        def mapped(address, size, executable=False):
            matches = []
            for p in loads:
                delta = address - p[3]
                if 0 <= delta < p[5] and size <= p[5] - delta:
                    if executable and not p[1] & 1:
                        raise ValueError("ELF export is not executable")
                    matches.append(p[2] + delta)
            if len(matches) != 1:
                raise ValueError("ELF export address is not uniquely file-backed")
            return matches[0]

        dynamic = [p for p in programs if p[0] == 2]
        if len(dynamic) != 1 or not dynamic[0][5] or dynamic[0][5] % 16:
            raise ValueError("invalid ELF export dynamic segment")
        dynamic = dynamic[0]
        if mapped(dynamic[3], dynamic[5]) != dynamic[2]:
            raise ValueError("ELF export dynamic segment mapping differs")
        tags = {}
        lookup_tags = {4, 5, 6, 10, 11, 0x6ffffef5, 0x6ffffff0}
        for offset in range(dynamic[2], dynamic[2] + dynamic[5], 16):
            tag, value = unpack("<qQ", offset)
            if tag == 0:
                break
            if tag in lookup_tags:
                if tag in tags:
                    raise ValueError("duplicate ELF export dynamic tag")
                tags[tag] = value
        else:
            raise ValueError("unterminated ELF export dynamic segment")

        def runtime_table(tag, table):
            if tags.get(tag) != table[3] or mapped(table[3], table[5]) != table[4]:
                raise ValueError("ELF export section differs from runtime table")

        runtime_table(6, symbols)
        runtime_table(5, strings)
        if tags.get(11) != 24 or tags.get(10) != strings[5]:
            raise ValueError("ELF export runtime table size differs")
        if bool(versions) != (0x6ffffff0 in tags):
            raise ValueError("ELF export version table differs from runtime")
        if versions:
            runtime_table(0x6ffffff0, versions[0])

        hashes = {}
        for tag, kind in [(0x6ffffef5, 0x6ffffff6), (4, 5)]:
            tables = [s for s in sections if s[1] == kind]
            if len(tables) > 1 or bool(tables) != (tag in tags):
                raise ValueError("ELF export hash table differs from runtime")
            if tables:
                table = tables[0]
                if table[6] != sections.index(symbols):
                    raise ValueError("invalid ELF export hash symbol link")
                runtime_table(tag, table)
                hashes[tag] = table
        symbol_count = symbols[5] // 24

        def symbol_name(index):
            name = unpack("<I", symbols[4] + index * 24)[0]
            if name >= strings[5]:
                raise ValueError("ELF export name outside string table")
            return string(strings[4] + name, strings[4] + strings[5])

        if 0x6ffffef5 in hashes:
            table = hashes[0x6ffffef5]
            if table[5] < 16:
                raise ValueError("invalid ELF export GNU hash header")
            buckets, first, words, shift = unpack("<IIII", table[4])
            if not buckets or not words or words & (words - 1) or shift >= 32 or first > symbol_count:
                raise ValueError("invalid ELF export GNU hash header")
            bloom = table[4] + 16
            bucket = bloom + words * 8
            chain = bucket + buckets * 4
            if chain > table[4] + table[5]:
                raise ValueError("ELF export GNU hash outside table")

            # The loader does not use .dynsym's section size. Derive the last
            # reachable symbol from the runtime hash so a shortened inventory
            # cannot hide a trailing export. Starting at the largest bucket is
            # sufficient: every chain advances by one to its first terminator.
            last = 0
            for i in range(buckets):
                index = unpack("<I", bucket + i * 4)[0]
                if index and index < first:
                    raise ValueError("invalid ELF export GNU hash bucket")
                last = max(last, index)
            if last:
                offset = chain + (last - first) * 4
                while offset + 4 <= table[4] + table[5]:
                    value = unpack("<I", offset)[0]
                    last += 1
                    if value & 1:
                        break
                    offset += 4
                else:
                    raise ValueError("unterminated ELF export GNU hash chain")
                if last != symbol_count:
                    raise ValueError("ELF export GNU hash symbol count differs")
            # An all-zero bucket table has no reachable exports; its symoffset
            # does not determine the size of the un-hashed symbol prefix.

            def lookup(text):
                hashed = 5381
                for byte in text.encode("ascii"):
                    hashed = (hashed * 33 + byte) & 0xffffffff
                word = unpack("<Q", bloom + ((hashed // 64) & (words - 1)) * 8)[0]
                mask = (1 << (hashed % 64)) | (1 << ((hashed >> shift) % 64))
                if word & mask != mask:
                    return None
                index = unpack("<I", bucket + (hashed % buckets) * 4)[0]
                if index == 0:
                    return None
                while first <= index < symbol_count:
                    value = unpack("<I", chain + (index - first) * 4)[0]
                    if value | 1 == hashed | 1 and symbol_name(index) == text:
                        return index
                    if value & 1:
                        return None
                    index += 1
                raise ValueError("invalid ELF export GNU hash chain")
        elif 4 in hashes:
            table = hashes[4]
            buckets, count = unpack("<II", table[4])
            if not buckets or count != symbol_count or 8 + 4 * (buckets + count) > table[5]:
                raise ValueError("invalid ELF export SysV hash header")
            bucket = table[4] + 8
            chain = bucket + buckets * 4

            def lookup(text):
                hashed = 0
                for byte in text.encode("ascii"):
                    hashed = (hashed << 4) + byte
                    high = hashed & 0xf0000000
                    hashed = (hashed ^ (high >> 24)) & ~high
                index = unpack("<I", bucket + (hashed % buckets) * 4)[0]
                seen = set()
                while index:
                    if index >= count or index in seen:
                        raise ValueError("invalid ELF export SysV hash chain")
                    seen.add(index)
                    if symbol_name(index) == text:
                        return index
                    index = unpack("<I", chain + index * 4)[0]
                return None
        else:
            raise ValueError("missing ELF export runtime hash table")
        for index, offset in enumerate(range(symbols[4], symbols[4] + symbols[5], 24)):
            name, info, visibility, section, value, size = unpack("<IBBHQQ", offset)
            if name >= strings[5]:
                raise ValueError("ELF export name outside string table")
            # Linux/glibc also exposes GNU-unique definitions and interprets
            # visibility through the low two bits of st_other.
            if section and info >> 4 in (1, 2, 10) and visibility & 3 in (0, 3):
                text = string(strings[4] + name, strings[4] + strings[5])
                if text.startswith("fy_"):
                    # The pinned ABI uses unversioned exports. st_name alone
                    # also names non-default versions that dlsym cannot find.
                    if versions and unpack("<H", versions[0][4] + index * 2)[0] != 1:
                        raise ValueError("versioned FactsYOLO ELF export is unsupported")
                    if info & 15 != 2 or not 0 < section < section_count:
                        raise ValueError("FactsYOLO ELF export is not a defined function")
                    code = sections[section]
                    delta = value - code[3]
                    length = max(size, 1)
                    if (not value or code[1] != 1 or code[2] & 6 != 6
                            or delta < 0 or delta >= code[5] or length > code[5] - delta
                            or mapped(value, length, True) != code[4] + delta):
                        raise ValueError("ELF export is outside executable section")
                    if lookup(text) != index:
                        raise ValueError("ELF export is unreachable through runtime hash")
                    names.append(text)
                elif text and 0 < section < section_count and lookup(text) is not None:
                    # The package permits exactly the eleven named ABI exports,
                    # not only names with the fy_ prefix. dlsym can resolve
                    # STT_NOTYPE functions and globally visible data as well
                    # as STT_FUNC, so do not filter unexpected names by type.
                    # SHN_ABS version namespace labels are not file-backed
                    # definitions and remain outside this check.
                    raise ValueError("unexpected ELF named export")
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
        sections = [(*unpack("<IIII", start + i * 40 + 8), unpack("<I", start + i * 40 + 36)[0])
                    for i in range(section_count)]
        image_size = unpack("<I", optional + 56)[0]
        header_size = unpack("<I", optional + 60)[0]
        ranges = []
        for virtual_size, virtual, raw_size, raw, flags in sections:
            extent = virtual_size or raw_size
            if extent:
                if virtual < header_size or virtual + extent > image_size:
                    raise ValueError("PE export section outside image bounds")
                ranges.append((virtual, virtual + extent))
        ranges.sort()
        if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
            raise ValueError("overlapping PE export section mappings")

        def rva(address, size, executable=False):
            if address >= image_size or size > image_size - address:
                raise ValueError("PE export RVA outside image")
            matches = []
            for virtual_size, virtual, raw_size, raw, flags in sections:
                delta = address - virtual
                extent = min(virtual_size or raw_size, raw_size)
                if 0 <= delta < extent and size <= extent - delta:
                    span(raw, raw_size)
                    if executable and not flags & 0x20000000:
                        raise ValueError("PE export is not executable")
                    matches.append((raw + delta, raw + min(extent, image_size - virtual)))
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
        previous = None
        named_ordinals = set()
        for i in range(count):
            pointer, end = rva(unpack("<I", name_table + 4 * i)[0], 1)
            text = string(pointer, end)
            if previous is not None and previous >= text:
                raise ValueError("PE export names are not strictly sorted")
            previous = text
            # The package promises exactly the public ABI names, not merely
            # the expected fy_* subset. A named helper or alias is callable
            # through GetProcAddress even if its ordinal is already named.
            if not text.startswith("fy_"):
                raise ValueError("unexpected PE named export")
            ordinal = unpack("<H", ordinal_table + 2 * i)[0]
            if ordinal >= functions:
                raise ValueError("PE export ordinal outside address table")
            named_ordinals.add(ordinal)
            address = unpack("<I", address_table + 4 * ordinal)[0]
            if text.startswith("fy_"):
                if not address or directory <= address < directory + directory_size:
                    raise ValueError("FactsYOLO PE export is absent or forwarded")
                rva(address, 1, True)
                names.append(text)
        # GetProcAddress accepts ordinal-only exports omitted from the sorted
        # name table. The full nonzero address table must be accounted for.
        for ordinal in range(functions):
            address = unpack("<I", address_table + 4 * ordinal)[0]
            if address and ordinal not in named_ordinals:
                raise ValueError("unexpected PE ordinal-only export")
    else:
        raise ValueError("unsupported native export RID")
    if len(names) != len(set(names)):
        raise ValueError("duplicate FactsYOLO native exports")
    return sorted(names)
