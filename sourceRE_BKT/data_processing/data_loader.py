import pandas as pd
import numpy as np

class DataLoader:
    """Maneja la carga, limpieza y estandarización de ASSISTments."""
    
    def __init__(self):
        # Columnas estándar que requiere el clustering y el ModelBase
        self.target_emotions = ['frustrated', 'confused', 'concentrating', 'bored']

    def clean_2012(self, path):
        df = pd.read_csv(path, encoding="ISO-8859-15", low_memory=False)
        # Renombrado 2012
        mapping = {
            'Average_confidence(FRUSTRATED)': 'frustrated',
            'Average_confidence(CONFUSED)': 'confused',
            'Average_confidence(CONCENTRATING)': 'concentrating',
            'Average_confidence(BORED)': 'bored',
            'start_time': 'start_time',
            'end_time': 'end_time'
        }
        df.rename(columns=mapping, inplace=True)
        irrelevant = ['problem_id','assignment_id','problem_type','original','bottom_hint','answer_id','answer_text',
                      'hint_count','actions','ms_first_response','tutor_mode','student_class_id','position','type',
                      'base_sequence_id','overlap_time','teacher_id','school_id', 'problemlogid','first_action',
                      'assistment_id', 'sequence_id', 'template_id',]
        
        df.drop(columns=[c for c in irrelevant if c in df.columns], inplace=True)
        return self._base_clean(df, 'start_time', 'end_time')

    def clean_2017(self, path):
        df = pd.read_csv(path, encoding="ISO-8859-15", low_memory=False)
        # Renombrado 2017
        mapping = {
            'RES_FRUSTRATED': 'frustrated',
            'RES_CONFUSED': 'confused',
            'RES_CONCENTRATING': 'concentrating',
            'RES_BORED': 'bored',
            'studentId': 'user_id',
            'startTime': 'start_time',
            'endTime': 'end_time'
        }
        df.rename(columns=mapping, inplace=True)
        
        irrelevant = ['SY ASSISTments Usage','AveCarelessness','AveResOfftask','AveResGaming',
                      'problemId','assignmentId','assistmentId','original','hint','hintTotal','scaffold',
                      'bottomHint','frIsHelpRequest','frPast5HelpRequest','frPast8HelpRequest','stlHintUsed',
                      'past8BottomOut','totalFrPercentPastWrong','totalFrPastWrongCount','frPast5WrongCount',
                      'frPast8WrongCount','totalFrTimeOnSkill','frWorkingInSchool','totalFrAttempted','responseIsFillIn',
                      'responseIsChosen','endsWithScaffolding','endsWithAutoScaffolding','frTimeTakenOnScaffolding',
                      'frTotalSkillOpportunitiesScaffolding','totalFrSkillOpportunitiesByScaffolding',
                      'frIsHelpRequestScaffolding','timeGreater5Secprev2wrong','sumRight','helpAccessUnder2Sec',
                      'timeGreater10SecAndNextActionRight','consecutiveErrorsInRow','sumTime3SDWhen3RowRight',
                      'sumTimePerSkill','totalTimeByPercentCorrectForskill','timeOver80','manywrong',
                      'confidence(BORED)','confidence(CONCENTRATING)','confidence(CONFUSED)','confidence(FRUSTRATED)',
                      'confidence(OFF TASK)','confidence(GAMING)','RES_OFFTASK','RES_GAMING', 'Prev5count',
                      'InferredGender','Selective','isSTEM','Enrolled','MiddleSchoolId',
                      'AveKnow', 'AveCorrect', 'NumActions', 'action_num', 'problemType', 'timeTaken', 'hintCount',
                      'timeSinceSkill', 'totalFrSkillOpportunities', 'MCAS', 'Ln-1', 'Ln', 'AveResBored',
                      'AveResEngcon','AveResConf','AveResFrust']
        
        df.drop(columns=[c for c in irrelevant if c in df.columns], inplace=True)
        
        return self._base_clean(df, 'start_time', 'end_time')

    def _base_clean(self, df, start_col, end_col):
        """Lógica de limpieza: intercambia fechas inconsistentes y ordena cronológicamente."""
        df[start_col] = pd.to_datetime(df[start_col], format='mixed')
        df[end_col] = pd.to_datetime(df[end_col], format='mixed')
    
        # Si start_time > end_time, los intercambiamos
        mask = df[start_col] > df[end_col]
        if mask.any():
            print(f"Se intercambiaron {mask.sum()} registros con fechas invertidas.")
            df.loc[mask, [start_col, end_col]] = df.loc[mask, [end_col, start_col]].values # Se usa.values para evitar conflictos de índices durante el intercambio

        # Se ordena por habilidad, luego por usuario y finalmente por tiempo de inicio
        df = df.sort_values(by=['skill', 'user_id', start_col])
        df['correct'] = df['correct'].astype(int)
        
        return df.dropna(subset=['skill', 'user_id', 'concentrating'])