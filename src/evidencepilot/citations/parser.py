"""Markdown-aware atomic claim parsing."""

from markdown_it import MarkdownIt

from ..models import AtomicClaim


class CitationParserMixin:
    @staticmethod
    def _inline_source_ids(text: str) -> list[str]:
        """Read citation labels while respecting Markdown code and link syntax."""
        source_ids: list[str] = []
        index = 0
        code_fence = 0
        while index < len(text):
            if text[index] == "`":
                run_end = index
                while run_end < len(text) and text[run_end] == "`":
                    run_end += 1
                run = run_end - index
                if code_fence == 0:
                    code_fence = run
                elif run == code_fence:
                    code_fence = 0
                index = run_end
                continue
            if code_fence or text[index] != "[" or (index and text[index - 1] == "\\"):
                index += 1
                continue
            close = text.find("]", index + 1)
            if close < 0:
                break
            label = text[index + 1:close]
            # A Markdown link whose label happens to be S1 is not a citation marker.
            if close + 1 < len(text) and text[close + 1] == "(":
                index = close + 1
                continue
            if len(label) > 1 and label[0] == "S" and label[1:].isdigit():
                source_id = label
                if source_id not in source_ids:
                    source_ids.append(source_id)
            index = close + 1
        return source_ids

    @staticmethod
    def _sentence_spans(text: str) -> list[tuple[int, int]]:
        """Return sentence spans without flattening Markdown inline markup."""
        spans: list[tuple[int, int]] = []
        start = 0
        index = 0
        code_fence = 0
        while index < len(text):
            char = text[index]
            if char == "`":
                run_end = index
                while run_end < len(text) and text[run_end] == "`":
                    run_end += 1
                run = run_end - index
                if code_fence == 0:
                    code_fence = run
                elif run == code_fence:
                    code_fence = 0
                index = run_end
                continue
            boundary = not code_fence and char in "。！？!?"
            if not code_fence and char == ".":
                after = index + 1
                while after < len(text) and text[after].isspace():
                    after += 1
                boundary = after > index + 1 and (
                    after == len(text) or text[after].isupper() or text[after].isdigit()
                    or text[after] in "*_`["
                )
            if boundary:
                end = index + 1
                left = start
                while left < end and text[left].isspace():
                    left += 1
                if left < end:
                    spans.append((left, end))
                start = end
                while start < len(text) and text[start].isspace():
                    start += 1
                index = start
                continue
            index += 1
        left = start
        while left < len(text) and text[left].isspace():
            left += 1
        end = len(text)
        while end > left and text[end - 1].isspace():
            end -= 1
        if left < end:
            spans.append((left, end))
        return spans

    @staticmethod
    def _atomic_claims(report: str) -> list[AtomicClaim]:
        claims: list[AtomicClaim] = []
        parser = MarkdownIt("commonmark").enable("table")
        tokens = parser.parse(report)
        lines = report.splitlines(keepends=True)
        line_offsets = [0]
        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line))

        stack: list[str] = []
        references_level: int | None = None
        heading_level: int | None = None
        heading_text = ""
        reference_names = {"references", "reference", "参考文献", "参考资料", "来源", "sources"}

        for token in tokens:
            if token.type == "heading_open":
                heading_level = int(token.tag[1:])
                stack.append(token.type)
                continue
            if token.type == "heading_close":
                normalized = heading_text.strip().rstrip(":：").casefold()
                if normalized in reference_names:
                    references_level = heading_level
                elif references_level is not None and heading_level is not None and heading_level <= references_level:
                    references_level = None
                heading_level = None
                heading_text = ""
                if stack and stack[-1] == "heading_open":
                    stack.pop()
                continue
            if token.type == "tr_open" and "tbody_open" in stack and token.map is not None:
                start_line, end_line = token.map
                block_start = line_offsets[start_line]
                block_end = line_offsets[min(end_line, len(line_offsets) - 1)]
                raw_row = report[block_start:block_end]
                row_text = raw_row.strip().strip("|").strip()
                row_start = raw_row.find(row_text)
                cell_cursor = 0
                for raw_cell in row_text.split("|"):
                    cell = raw_cell.strip()
                    cell_start = row_text.find(cell, cell_cursor)
                    cell_cursor = cell_start + len(cell)
                    for sentence_start, sentence_end in CitationParserMixin._sentence_spans(cell):
                        sentence = cell[sentence_start:sentence_end]
                        source_ids = CitationParserMixin._inline_source_ids(sentence)
                        start_offset = block_start + row_start + cell_start + sentence_start
                        claims.append(AtomicClaim(
                            claim_id=f"CL{len(claims) + 1}", text=sentence,
                            source_ids=source_ids, node_type="table_cell",
                            start_offset=start_offset, end_offset=start_offset + len(sentence),
                            start_line=start_line, end_line=max(start_line, end_line - 1),
                        ))
            if token.nesting == 1:
                stack.append(token.type)
                continue
            if token.nesting == -1:
                opening = token.type.removesuffix("_close") + "_open"
                if opening in stack:
                    stack.remove(opening)
                continue
            if token.type != "inline":
                continue
            if "td_open" in stack or "th_open" in stack:
                continue
            if "heading_open" in stack:
                heading_text = token.content
                continue
            if references_level is not None or token.map is None:
                continue

            start_line, end_line = token.map
            block_start = line_offsets[start_line]
            block_end = line_offsets[min(end_line, len(line_offsets) - 1)]
            raw_block = report[block_start:block_end]
            inline_start = raw_block.find(token.content)
            if inline_start < 0:
                continue
            if "td_open" in stack or "th_open" in stack:
                node_type = "table_cell"
            elif "list_item_open" in stack:
                node_type = "list_item"
            elif "blockquote_open" in stack:
                node_type = "blockquote"
            else:
                node_type = "paragraph"
            for local_start, local_end in CitationParserMixin._sentence_spans(token.content):
                sentence = token.content[local_start:local_end]
                source_ids = CitationParserMixin._inline_source_ids(sentence)
                start_offset = block_start + inline_start + local_start
                end_offset = start_offset + len(sentence)
                claims.append(AtomicClaim(
                    claim_id=f"CL{len(claims) + 1}", text=sentence,
                    source_ids=source_ids, node_type=node_type,
                    start_offset=start_offset, end_offset=end_offset,
                    start_line=report.count("\n", 0, start_offset),
                    end_line=report.count("\n", 0, end_offset),
                ))
        return claims




def parse_atomic_claims(report: str) -> list[AtomicClaim]:
    return CitationParserMixin._atomic_claims(report)
