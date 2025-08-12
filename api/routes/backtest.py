from fastapi import APIRouter
router = APIRouter()

@router.post('/backtest')
def backtest_portfolio(data: dict):
    return {'backtest': True}
