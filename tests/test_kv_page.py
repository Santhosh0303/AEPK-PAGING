import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from aepk_paging.kv_page import KVPage, PageTable, ResidencyTier


@st.composite
def kv_arrays(draw):
    rows = draw(st.integers(min_value=1, max_value=4))
    cols = draw(st.integers(min_value=1, max_value=4))
    values = st.integers(min_value=-1000, max_value=1000)
    k_values = draw(st.lists(values, min_size=rows * cols, max_size=rows * cols))
    v_values = draw(st.lists(values, min_size=rows * cols, max_size=rows * cols))
    K = np.array(k_values, dtype=np.int32).reshape((rows, cols))
    V = np.array(v_values, dtype=np.int32).reshape((rows, cols))
    return K, V


@given(kv=kv_arrays(), layer=st.integers(min_value=0, max_value=3))
def test_resident_store_fetch_round_trip_is_bitwise_identical(kv, layer) -> None:
    K, V = kv
    table = PageTable()
    page = KVPage(
        page_id="page-0",
        layer=layer,
        token_range=(0, K.shape[0]),
        K=K,
        V=V,
        precision_tag="int32",
        attention_mass=1.0,
    )

    physical_id = table.store(page, tier=ResidencyTier.RESIDENT)
    fetched = table.fetch("page-0")

    assert table.entry("page-0").physical_id == physical_id
    assert table.entry("page-0").tier is ResidencyTier.RESIDENT
    assert np.array_equal(fetched.K, K)
    assert np.array_equal(fetched.V, V)
    table.validate_invariants()


@given(ops=st.lists(st.sampled_from(["store", "fetch", "delete"]), min_size=1, max_size=30))
def test_page_table_invariants_hold_across_random_op_sequences(ops) -> None:
    table = PageTable()
    live_ids: set[int] = set()
    next_id = 0

    for op in ops:
        if op == "store":
            page_id = next_id
            next_id += 1
            base = np.array([[page_id, page_id + 1]], dtype=np.int32)
            table.store(
                KVPage(
                    page_id=page_id,
                    layer=0,
                    token_range=(page_id * 2, page_id * 2 + 2),
                    K=base,
                    V=base + 10,
                    precision_tag="int32",
                    attention_mass=float(page_id + 1),
                ),
                tier=ResidencyTier.RESIDENT,
            )
            live_ids.add(page_id)
        elif op == "fetch" and live_ids:
            page_id = min(live_ids)
            fetched = table.fetch(page_id)
            assert fetched.page_id == page_id
            assert np.array_equal(fetched.K, np.array([[page_id, page_id + 1]], dtype=np.int32))
        elif op == "delete" and live_ids:
            page_id = min(live_ids)
            table.delete(page_id)
            live_ids.remove(page_id)

        table.validate_invariants()
