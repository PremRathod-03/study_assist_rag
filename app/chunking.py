from langchain_text_splitters import RecursiveCharacterTextSplitter

# This splitter tries to break text at natural boundaries first
# (paragraph breaks, then sentences, then words) before resorting
# to a hard cutoff — so it rarely slices a sentence in half.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # roughly how many characters per chunk
    chunk_overlap=100,   # overlap so context isn't lost at chunk edges
)


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Takes the output of loaders.load_file() (one dict per page/slide)
    and splits each page's text into smaller chunks.
    Keeps the original metadata (source_type, page/slide number) attached
    to every chunk that came from that page, so we never lose track of
    where a chunk originally came from.
    """
    all_chunks = []

    for page in pages:
        pieces = splitter.split_text(page["text"])
        for piece in pieces:
            all_chunks.append({
                "text": piece,
                "source_type": page["source_type"],
                "page": page.get("page", page.get("slide")),
            })

    return all_chunks