"""Estimation's rules in JAX: whole batches of games stepped together on the GPU.

``rules`` is one round as pure array functions (``init_round``, ``legal_mask``, ``step``) and
``game`` chains rounds into a Bola. Both mirror Yosif Elessawi's engine rule for rule; the
tests in tests/test_jaxenv.py replay thousands of games in both and require identical legal
moves, turns and scores at every step. Only the engine's default rules are supported.
"""
