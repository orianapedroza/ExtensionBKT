# Contenido de sourceRE-BKT/models/__init__.py
from .emotion_internal.re_bkt import REBKT as REBKT_Internal
from .emotion_external.re_bkt import REBKT as REBKT_External
from .emotion_hybrid.re_bkt import REBKT as REBKT_Hybrid

__all__ = ['REBKT_Internal', 'REBKT_External', 'REBKT_Hybrid']

# Importación limpia gracias a los __init__.py
#from models import REBKT_Internal, REBKT_External, REBKT_Hybrid
#from fit.model_base import ModelBase #

# Evaluar el modelo híbrido (Versión 3)
#modelo_hibrido = ModelBase(model_class=REBKT_Hybrid)
#modelo_hibrido.fit(train_df) #